"""Downloads that survive a flaky connection, for the Paranoia route (docs/RUN_ON_YOUR_GPU.md, route 4 moves about 2.9 GB). Each file
goes to DST.part, resumes there with an HTTP Range request after a dropped connection, and becomes DST only when the transfer ends
cleanly; so an interrupted run never leaves a half file under the final name. Retries with backoff; a 4xx answer other than 408 and
429 is final (a private release answers 404, and retrying cannot help). The callers still check every file's size or SHA-256.
Standard library only."""
import http.client
import os
import socket
import ssl
import time
import urllib.error
import urllib.request

TRIES = 8
CHUNK = 1 << 20


def expected_total(r, offset: int):
    """The whole file's size the answer announces (Content-Range's total, else offset + Content-Length), or None if it does not say."""
    cr = r.headers.get("Content-Range") if hasattr(r, "headers") else None
    if cr and "/" in cr and cr.rsplit("/", 1)[1].isdigit():
        return int(cr.rsplit("/", 1)[1])
    n = r.headers.get("Content-Length") if hasattr(r, "headers") else None
    return offset + int(n) if n and n.isdigit() else None


def range_start(r):
    """The first byte a 206 answer carries (its Content-Range), or None."""
    cr = r.headers.get("Content-Range", "") if hasattr(r, "headers") else ""
    try:
        return int(cr.split()[1].split("-")[0])
    except (IndexError, ValueError):
        return None


def get(url: str, dst: str, tries: int = TRIES, timeout: float = 60, sleep=time.sleep, log=print) -> str:
    part = dst + ".part"
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if url.startswith("file:"):                                  # a local folder: nothing to wait for, a missing file is final
        tries = 1
    for attempt in range(1, tries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                # a resume only if the server answers 206 from exactly our offset; a 200 (range ignored) or another offset restarts
                resumed = have > 0 and getattr(r, "status", None) == 206 and range_start(r) == have
                if getattr(r, "status", None) == 206 and not resumed:     # a part of the file we did not ask for: start over
                    if os.path.exists(part):
                        os.remove(part)
                    raise ConnectionError(f"the server answered from byte {range_start(r)}, not {have}")
                total = expected_total(r, have if resumed else 0)
                with open(part, "ab" if resumed else "wb") as f:
                    while True:
                        b = r.read(CHUNK)
                        if not b:
                            break
                        f.write(b)
            got = os.path.getsize(part)
            if total is not None and got < total:          # http.client ends a dropped transfer quietly: a short file is a drop too
                raise ConnectionError(f"connection closed at {got:,} of {total:,} bytes")
            os.replace(part, dst)
            return dst
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:                    # no bytes at our offset: whole if the server's total says so, else start over
                total = (e.headers.get("Content-Range") or "").rsplit("/", 1)[-1] if e.headers else ""
                if total.isdigit() and int(total) == have:
                    os.replace(part, dst)
                    return dst
                os.remove(part)
            elif 400 <= e.code < 500 and e.code not in (408, 429):
                raise
            err = e
        except (urllib.error.URLError, ConnectionError, http.client.HTTPException, socket.timeout, TimeoutError, ssl.SSLError) as e:
            err = e
        if attempt == tries:
            raise err
        wait = min(2 ** attempt, 60)
        log(f"{os.path.basename(dst)}: {type(err).__name__}; retry {attempt} of {tries - 1} in {wait} s"
            f" (resuming at {os.path.getsize(part) if os.path.exists(part) else 0:,} bytes)", flush=True)
        sleep(wait)
    raise AssertionError("unreachable")
