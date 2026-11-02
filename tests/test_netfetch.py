"""scripts/netfetch.py against a local server that drops connections, ignores ranges or answers 404 (a judge's route 4 moves about
2.9 GB; the first laptop run of it died on a connection reset). No network."""
import http.server
import os
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import netfetch  # noqa: E402

BODY = bytes(range(256)) * 4096                                     # 1 MiB


def serve(behaviour: dict):
    """A server for /f: drops the first `drops` connections after `cut` bytes, honours Range unless ranges is False; /missing is 404."""
    calls = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            calls.append(self.headers.get("Range"))
            if self.path != "/f":
                self.send_error(404); return
            start = 0
            rng = self.headers.get("Range")
            if rng and behaviour.get("ranges", True):
                start = int(rng.split("=")[1].split("-")[0])
                self.send_response(206); self.send_header("Content-Range", f"bytes {start}-{len(BODY) - 1}/{len(BODY)}")
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(BODY) - start)); self.end_headers()
            if behaviour.get("drops", 0) > 0:
                behaviour["drops"] -= 1
                self.wfile.write(BODY[start:start + behaviour["cut"]]); self.wfile.flush()
                self.connection.shutdown(2); return                     # the connection drops mid-file
            self.wfile.write(BODY[start:])

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}", calls


def quiet(*a, **k):
    pass


def test_resumes_after_dropped_connections(tmp_path):
    srv, base, calls = serve({"drops": 2, "cut": 300_000})
    try:
        dst = tmp_path / "d" / "f.bin"
        netfetch.get(base + "/f", str(dst), sleep=lambda s: None, log=quiet)
    finally:
        srv.shutdown()
    assert dst.read_bytes() == BODY and not (tmp_path / "d" / "f.bin.part").exists()
    assert calls[0] is None and calls[1] == "bytes=300000-" and calls[2] == "bytes=600000-"   # each retry asks only for the rest


def test_a_server_that_ignores_ranges_restarts_the_file(tmp_path):
    srv, base, calls = serve({"drops": 1, "cut": 300_000, "ranges": False})
    try:
        dst = tmp_path / "f.bin"
        netfetch.get(base + "/f", str(dst), sleep=lambda s: None, log=quiet)
    finally:
        srv.shutdown()
    assert dst.read_bytes() == BODY                                   # not the first 300,000 bytes twice


def test_a_404_is_final_and_leaves_no_file(tmp_path):
    srv, base, calls = serve({})
    waits = []
    try:
        with pytest.raises(Exception) as e:
            netfetch.get(base + "/missing", str(tmp_path / "x"), sleep=waits.append, log=quiet)
    finally:
        srv.shutdown()
    assert "404" in str(e.value) and waits == [] and len(calls) == 1 and not (tmp_path / "x").exists()


def test_gives_up_after_its_tries_without_a_final_file(tmp_path):
    srv, base, calls = serve({"drops": 99, "cut": 1000})
    waits = []
    try:
        with pytest.raises(Exception):
            netfetch.get(base + "/f", str(tmp_path / "f.bin"), tries=3, sleep=waits.append, log=quiet)
    finally:
        srv.shutdown()
    assert len(calls) == 3 and waits == [2, 4] and not (tmp_path / "f.bin").exists()
    assert os.path.getsize(tmp_path / "f.bin.part") == 3000            # kept, so the next run resumes from it
