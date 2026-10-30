"""Vercel entry point for the hosted Pilot app (Vercel's Python runtime, file-based function in /api, top-level WSGI `app`; vercel.json
rewrites every path here and the app routes on the original path). The same `app` runs locally with `python web/serve.py`."""
import sys
from pathlib import Path

if sys.version_info < (3, 12):                             # scripts/build_pilot_page.py (imported by the guard) needs 3.12+
    raise RuntimeError("the StillHere web app needs Python 3.12 or newer")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web.app import app  # noqa: E402,F401
