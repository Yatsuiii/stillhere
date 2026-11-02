"""The worker's display name (cloud review of Nov 2, 12:15 UTC, item 4): what the page shows while a job runs and what the result label
names. A judge's local run says "local GPU" unless STILLHERE_WORKER_LABEL says otherwise; the laptop's service sets "the laptop"."""
import os
import re

_raw = os.environ.get("STILLHERE_WORKER_LABEL", "local GPU").strip()
WORKER_LABEL = _raw if re.fullmatch(r"[A-Za-z0-9 ._()-]{1,40}", _raw) else "local GPU"
