"""StillHere live worker (cloud plan v2, item 5; docs/worker.md, docs/jobs_api.md). Runs on the laptop inside the sandboxed service
(worker/stillhere-worker.service): pulls jobs from the page's backend over HTTPS (the laptop accepts no inbound connection), one at a
time, TRIBE v2 loaded once (worker/engine.py), posts real stage events and uploads region values only. A job is a recipe and nothing
else ({clip_id, condition, seed}, worker/recipe.py); anything outside the schema, any clip not on data/live_clips.json, and any job
whose cache key the worker does not reproduce is refused without running. Its only secret is its job token.

  serve       pull and run jobs until stopped (STILLHERE_JOBS_URL; token from $CREDENTIALS_DIRECTORY/job_token or --token-file)
  precompute  the 12 clips x 4 conditions at the default seed into data/forecast_cache.json (resumable; same engine as serve)
  self-test   from inside the service: hidden paths must not be listable, model folders must be read-only; prints PASS or FAIL lines
  once        run one recipe locally and print its timings (no network)
  run         one job in the foreground (--clip --condition --seed [--out f.json]): no hours, token or network; writes the region
              values, and checks them against data/forecast_cache.json when the recipe is there
  verify      4 cached recipes (one per condition) in one process: a table against data/forecast_cache.json; exit 1 if any is out of
              tolerance (tests/test_worker_gpu.py's: max abs difference 0.002, r 0.9999)
Local runs: STILLHERE_MODELS_DIR (scripts/judge_gpu_setup.sh's <models_dir>/worker) sets HF_HUB_CACHE and HF_HUB_OFFLINE=1;
CUDA_VISIBLE_DEVICES defaults to 0 (accelerate would spread Llama over every visible GPU); STILLHERE_WORKER_LABEL names the worker on
the page and in the result label (default "local GPU"; the laptop's service sets "the laptop").
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
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")                 # before torch loads: accelerate spreads Llama over every visible GPU
if os.environ.get("STILLHERE_MODELS_DIR"):                           # a judge's models (scripts/judge_gpu_setup.sh), read offline
    os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["STILLHERE_MODELS_DIR"]) / "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
from worker import recipe as R  # noqa: E402
from worker.labels import WORKER_LABEL  # noqa: E402

POLL_S = float(os.environ.get("STILLHERE_POLL_S", "2"))
HOURS_ET = os.environ.get("STILLHERE_HOURS_ET", "10:00-14:00")     # worker hours, America/New_York; "00:00-24:00" = any time
EXIT_RESTART = 75                                                  # non-zero: systemd (Restart=on-failure) starts a fresh worker
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
        return self.post("/api/worker/claim", {"worker": WORKER_LABEL, "worker_label": WORKER_LABEL, "models": models})

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


def within_hours(now=None, hours: str = None) -> bool:
    """True inside the worker hours (America/New_York wall clock, start inclusive, end exclusive)."""
    import datetime
    from zoneinfo import ZoneInfo
    now = now or datetime.datetime.now(ZoneInfo("America/New_York"))
    a, b = (hours or HOURS_ET).split("-")
    mins = lambda hm: int(hm.split(":")[0]) * 60 + int(hm.split(":")[1])
    return mins(a) <= now.hour * 60 + now.minute < mins(b)


def serve(args):
    """Polls for jobs inside the worker hours. Exit 0 (stay down until the timer's next start) once outside them with no job running;
    exit EXIT_RESTART after a job that filled TRIBE's feature cache, so systemd restarts it with an empty cache."""
    from worker.engine import Engine
    if not within_hours():
        log(f"outside worker hours ({HOURS_ET} ET); not starting"); return 0
    index = R.load_index(); jobs = Jobs(os.environ["STILLHERE_JOBS_URL"], read_token(args.token_file))
    engine = Engine(STATE / "feature_cache").load(); models = engine.models()
    log(f"engine loaded in {engine.load_s} s on {engine.gpu}; polling every {POLL_S} s until {HOURS_ET.split('-')[1]} ET")
    backoff = POLL_S
    while True:
        if not within_hours():
            log("worker hours over; stopping (exit 0)"); return 0
        try:
            code, body = jobs.claim(models)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            code, body = None, None
            log(f"claim failed ({type(e).__name__}); retrying in {backoff:.0f} s")
        if code == 200 and body and body.get("job"):
            backoff = POLL_S
            handle(body["job"], engine, jobs, index)
            if engine.cache_full():                                   # restart fresh between jobs rather than prune live
                log("feature cache past its limit; exiting for a fresh restart"); return EXIT_RESTART
            continue
        if code is not None and code not in (200, 204):
            log(f"claim answered {code}")
        # --max-polls (tests only; 0 = forever) counts every poll that brought no job, failed claims included
        if args.max_polls and args.max_polls > 0:
            args.max_polls -= 1
            if args.max_polls == 0:
                return 0
        if code is None:
            time.sleep(backoff); backoff = min(backoff * 2, 60)
        else:
            backoff = POLL_S; time.sleep(POLL_S)


def precompute(args):
    """The 12 clips x 4 conditions at the default seed, plus any extra recipes (--extra clip:condition:seed, kept in the file's
    extra_recipes, e.g. the page's example chips), into one cache file. Resumable; same engine as live jobs."""
    from worker.engine import Engine
    index = R.load_index(); out = Path(args.out)
    cache = json.loads(out.read_text()) if out.exists() else {}
    entries = cache.get("entries", {}); extras = cache.get("extra_recipes", [])
    for spec in args.extra or []:
        c, cond, seed = spec.split(":"); rec = R.normalize({"clip_id": c, "condition": cond, "seed": int(seed)}, index)
        if rec not in extras:
            extras.append(rec)
    todo = [{"clip_id": c["id"], "condition": cond, "seed": 0 if cond in R.SEEDLESS else index["default_seed"]}
            for c in index["clips"] for cond in R.CONDITIONS] + extras
    engine = None
    for i, rec in enumerate(todo, 1):
        key = R.cache_key(rec, index)
        if key in entries:
            continue
        engine = engine or Engine(STATE / "feature_cache").load()
        res = engine.run(R.build_condition(rec, index), STATE / "jobs", lambda *a: None)
        entries[key] = {"cache_key": key, "recipe": rec, **res}
        doc = {"about": "Cached TRIBE v2 forecasts for the live page (docs/worker.md): the 12 clips x 4 conditions at the default seed, "
                        "plus extra_recipes, computed by worker/tribe_worker.py precompute with the same engine as live jobs. Region "
                        "values only. Forecast, not a scan.",
               "models": index["models"], "default_seed": index["default_seed"], "extra_recipes": extras, "entries": entries}
        tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(doc, indent=1) + "\n"); tmp.replace(out)
        log(f"{i}/{len(todo)} {rec['clip_id']} {rec['condition']} seed {rec['seed']}: {res['timings_s']}")


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


TOL_MAXABS, TOL_R = 0.002, 0.9999                                   # tests/test_worker_gpu.py's, from the repeats in docs/worker.md


def compare(rec: dict, res: dict, index: dict, cache_path=HERE.parent / "data/forecast_cache.json"):
    """This run against our cached forecast of the same recipe: {region: (max abs difference, r)} and ok, or None if not cached."""
    import numpy as np
    entries = json.loads(Path(cache_path).read_text()).get("entries", {})
    c = entries.get(R.cache_key(rec, index))
    if c is None:
        return None
    out = {}
    for k in ("language", "hearing"):
        x, y = np.array(res["regions"][k], float), np.array(c["regions"][k], float)
        if x.shape != y.shape:
            out[k] = (float("inf"), float("nan")); continue
        out[k] = (float(np.max(np.abs(x - y))), float(np.corrcoef(x, y)[0, 1]))
    ok = all(m <= TOL_MAXABS and r >= TOL_R for m, r in out.values()) and res.get("n_words_text_lane") == c.get("n_words_text_lane")
    return {"regions": out, "n_words": (res.get("n_words_text_lane"), c.get("n_words_text_lane")), "ok": ok}


def cmp_line(cmp) -> str:
    if cmp is None:
        return "not in data/forecast_cache.json: nothing to compare"
    parts = [f"{k} max abs {m:.4f}, r {r:.5f}" for k, (m, r) in cmp["regions"].items()]
    return ("PASS" if cmp["ok"] else "FAIL") + f" against our cached forecast: {'; '.join(parts)}; words {cmp['n_words'][0]} vs {cmp['n_words'][1]}"


def run_one(args):
    """One job in the foreground on this machine's GPU: no worker hours, no token, no network."""
    from worker.engine import Engine
    index = R.load_index(); rec = R.normalize({"clip_id": args.clip, "condition": args.condition, "seed": args.seed}, index)
    engine = Engine(STATE / "feature_cache").load()
    log(f"engine loaded in {engine.load_s} s on {engine.gpu}")
    res = engine.run(R.build_condition(rec, index), STATE / "jobs", lambda st, lane, status, d: log(f"{st} {lane or ''} {status}"))
    res = {**res, "recipe": rec, "cache_key": R.cache_key(rec, index), "worker_label": WORKER_LABEL}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n"); log(f"region values written to {args.out}")
    log(f"{rec['clip_id']} {rec['condition']} seed {rec['seed']}: {res['seconds']} s of clip, total {res['timings_s']['total']} s, "
        f"peak {res.get('peak_vram_gb')} GB")
    cmp = compare(rec, res, index); log(cmp_line(cmp))
    return 1 if cmp is not None and not cmp["ok"] else 0


def verify_picks(entries: dict, index: dict) -> list:
    """One recipe per condition: the first clip (by id) cached at that condition's default seed."""
    picks = []
    for cond in R.CONDITIONS:
        recs = sorted((e["recipe"] for e in entries.values() if e["recipe"]["condition"] == cond
                       and e["recipe"]["seed"] == (0 if cond in R.SEEDLESS else index["default_seed"])), key=lambda r: r["clip_id"])
        picks.append(recs[0])
    return picks


def verify(args):
    """One cached recipe per condition (the first clip in data/forecast_cache.json at the default seed), in one process."""
    from worker.engine import Engine
    index = R.load_index()
    picks = verify_picks(json.loads((HERE.parent / "data/forecast_cache.json").read_text())["entries"], index)
    engine = Engine(STATE / "feature_cache").load()
    log(f"engine loaded in {engine.load_s} s on {engine.gpu}")
    rows, ok = [], True
    for rec in picks:
        res = engine.run(R.build_condition(rec, index), STATE / "jobs", lambda *a: None)
        cmp = compare(rec, res, index); ok &= bool(cmp and cmp["ok"])
        (l_m, l_r), (h_m, h_r) = cmp["regions"]["language"], cmp["regions"]["hearing"]
        rows.append((rec["condition"], rec["clip_id"], res["seconds"], l_m, l_r, h_m, h_r, res["timings_s"]["total"], res["peak_vram_gb"],
                     "PASS" if cmp["ok"] else "FAIL"))
    print(f"{'condition':18} {'clip':12} {'clip s':>6} {'lang max':>9} {'lang r':>8} {'hear max':>9} {'hear r':>8} {'job s':>6} {'peak GB':>7}  result")
    for r in rows:
        print(f"{r[0]:18} {r[1]:12} {r[2]:6d} {r[3]:9.4f} {r[4]:8.5f} {r[5]:9.4f} {r[6]:8.5f} {r[7]:6.1f} {r[8]!s:>7}  {r[9]}")
    print(f"tolerance: max abs {TOL_MAXABS}, r {TOL_R}; GPU {engine.gpu}; " + ("VERIFY PASS" if ok else "VERIFY FAIL"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve"); s.add_argument("--token-file"); s.add_argument("--max-polls", type=int, default=0)
    p = sub.add_parser("precompute"); p.add_argument("--out", default=str(HERE.parent / "data/forecast_cache.json"))
    p.add_argument("--extra", action="append", help="clip_id:condition:seed, may repeat")
    sub.add_parser("self-test")
    o = sub.add_parser("once"); o.add_argument("recipe")
    r = sub.add_parser("run"); r.add_argument("--clip", required=True); r.add_argument("--condition", required=True, choices=R.CONDITIONS)
    r.add_argument("--seed", type=int, default=None, help="default: the page's default seed (ignored for intact and reversed)")
    r.add_argument("--out", help="write the region values as JSON")
    sub.add_parser("verify")
    a = ap.parse_args(argv)
    if a.cmd == "run" and a.seed is None:
        a.seed = 0 if a.condition in R.SEEDLESS else R.load_index()["default_seed"]
    return {"serve": serve, "precompute": precompute, "self-test": self_test, "once": once, "run": run_one, "verify": verify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
