"""A stranger's fresh clone (cloud review of Nov 2, 12:15 UTC, item 1): `python web/serve.py` with an empty HOME and no keys answers
the default Black Cat PLAN with its recorded run, instead of refusing it for a missing question list. A home list, where one exists,
still applies (review of 14:35 UTC, item 1): the laptop's terms are not swapped for the hosted list's."""
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BLACK_CAT = ("Plan an exploratory stock TRIBE forecast of The Black Cat, intact versus sentences shuffled, in the language network.")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


def plan_in(home: Path, text: str) -> tuple:
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(home), "PYTHONPATH": os.environ.get("PYTHONPATH", "")}   # no keys
    port = free_port()
    proc = subprocess.Popen([sys.executable, "web/serve.py", "--port", str(port)], cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cookie = ""

    def call(path, body):
        nonlocal cookie
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Cookie": cookie})
        try:
            r = urllib.request.urlopen(req, timeout=60); code, raw, hdrs = r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            code, raw, hdrs = e.code, e.read(), e.headers
        if hdrs.get("Set-Cookie") and not cookie:
            cookie = hdrs["Set-Cookie"].split(";")[0]
        return code, json.loads(raw) if raw else {}

    try:
        for _ in range(150):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2); break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.2)
        call("/api/visitor", {})
        return call("/api/plan", {"text": text, "backend": "spark-1.3"})
    finally:
        proc.terminate(); proc.wait(10)


def test_fresh_home_without_keys_gets_the_recorded_plan(tmp_path):
    home = tmp_path / "home"; home.mkdir()
    code, d = plan_in(home, BLACK_CAT)
    assert code == 200 and d.get("label") == "recorded run", (code, {k: d.get(k) for k in ("label", "mode", "why", "error")})
    assert not any(home.iterdir()), "the local server wrote into HOME"


def test_a_home_list_still_applies(tmp_path):
    home = tmp_path / "home"; (home / ".config/stillhere").mkdir(parents=True)
    salt = bytes(range(32))
    (home / ".config/stillhere/question_blocklist.txt").write_text(f"salt {salt.hex()}\n" + hashlib.sha256(salt + b"quokkafield").hexdigest() + "\n")
    code, d = plan_in(home, BLACK_CAT.replace("The Black Cat", "The Black Cat quokkafield"))
    assert code == 200 and d.get("mode") == "refused", (code, d.get("mode"))       # the home list's term, not in the hosted list
    code, d = plan_in(home, BLACK_CAT)
    assert code == 200 and d.get("label") == "recorded run", (code, d.get("mode"))
