"""Test setup. stillhere/egress.py reads its question blocklist at import and fails closed without one. The laptop keeps its full list
outside the repository; a machine without it (CI) uses the committed server list, as the hosted app does, never a made-up one."""
import os
from pathlib import Path

if "STILLHERE_QUESTION_BLOCKLIST" not in os.environ and not (Path.home() / ".config/stillhere/question_blocklist.txt").is_file():
    os.environ["STILLHERE_QUESTION_BLOCKLIST"] = str(Path(__file__).resolve().parents[1] / "web/server_question_blocklist.txt")
