"""A local stand-in for the page's jobs API (docs/jobs_api.md), for the worker's tests and for an end-to-end run on this laptop: hands
out the queued jobs one per claim, records events, results and fails, and checks the token like the real server (401, no body).
usage: python3 worker/fake_jobs_server.py PORT TOKEN_FILE JOBS_JSON OUT_JSON    (serves until every job has a result or a fail)"""
import hmac
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class State:
    def __init__(self, token: str, jobs: list):
        self.token, self.queue, self.lock = token, list(jobs), threading.Lock()
        self.events, self.results, self.fails, self.claims, self.unauthorized = {}, {}, {}, 0, 0

    def done(self) -> bool:
        return not self.queue and all(j in self.results or j in self.fails for j in self.events)


def make_handler(st: State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, body=None):
            raw = json.dumps(body).encode() if body is not None else b""
            self.send_response(code); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

        def do_POST(self):
            auth = self.headers.get("Authorization", "")
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if not hmac.compare_digest(auth, f"Bearer {st.token}"):
                st.unauthorized += 1; return self._send(401)
            parts = self.path.strip("/").split("/")
            with st.lock:
                if self.path == "/api/worker/claim":
                    st.claims += 1
                    if not st.queue:
                        return self._send(204)
                    job = st.queue.pop(0); st.events[job["id"]] = []
                    return self._send(200, {"job": {**job, "claimed_utc": "now", "deadline_s": 40}})
                if len(parts) == 5 and parts[:3] == ["api", "worker", "jobs"]:
                    jid, what = parts[3], parts[4]
                    if what == "event":
                        st.events.setdefault(jid, []).append(body); return self._send(204)
                    if what == "result":
                        if jid in st.results or jid in st.fails:
                            return self._send(409)
                        st.results[jid] = body; return self._send(204)
                    if what == "fail":
                        st.fails[jid] = body.get("reason"); return self._send(204)
            return self._send(404)
    return H


def serve(port: int, st: State) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(st))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    import time
    port, token, jobs, out = int(sys.argv[1]), open(sys.argv[2]).read().strip(), json.load(open(sys.argv[3])), sys.argv[4]
    st = State(token, jobs); srv = serve(port, st)
    while not st.done() or st.claims == 0:
        time.sleep(1)
    srv.shutdown()
    json.dump({"claims": st.claims, "unauthorized": st.unauthorized, "events": st.events, "results": st.results, "fails": st.fails},
              open(out, "w"), indent=1)
    print(f"jobs done: {len(st.results)} results, {len(st.fails)} fails, {st.claims} claims, {st.unauthorized} unauthorized")
