"""Runs the hosted Pilot app locally with no account: the standard library's WSGI server (threaded), the in-memory store, and live
calls only where keys are in the environment (STILLHERE_LIVE=0 forces the recorded runs). Request lines are not logged.
usage: python web/serve.py [--port 8780]"""
import argparse
import os
import socketserver
import sys
from pathlib import Path
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# A fresh clone has no ~/.config/stillhere/question_blocklist.txt, and stillhere/egress.py refuses every typed request without a list,
# so PLAN, BUILD and FORECAST could not be reached. Only then (the variable unset and no home list) does the local server use the hosted
# server's list; the laptop's own home list, which holds terms the hosted list does not, still applies wherever it exists.
if not os.environ.get("STILLHERE_QUESTION_BLOCKLIST") and not (Path.home() / ".config/stillhere/question_blocklist.txt").is_file():
    os.environ["STILLHERE_QUESTION_BLOCKLIST"] = "web/server_question_blocklist.txt"
from web.app import app  # noqa: E402


class Quiet(WSGIRequestHandler):
    def log_message(self, *a):
        pass


class Threaded(socketserver.ThreadingMixIn, WSGIServer):
    daemon_threads = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8780); ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    with make_server(a.host, a.port, app, server_class=Threaded, handler_class=Quiet) as srv:
        print(f"StillHere Pilot (local) on http://{a.host}:{a.port}/", flush=True)
        srv.serve_forever()
