"""Vercel entry point: the same WSGI app (`app`) that web/serve.py runs locally."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from web.app import app  # noqa: E402,F401
