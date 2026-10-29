"""The server-side wording guard: every string a model writes (decline reasons, audit lines, anything shown from a model answer)
passes through `clean` before it leaves the server; a hit on the banned list shows a fixed template line instead. The list is the
page builder's list (scripts/build_pilot_page.py BANNED) plus the words the rulings of Oct 29 added for the live page."""
import importlib.util
import re
from pathlib import Path

_spec = importlib.util.spec_from_file_location("build_pilot_page", Path(__file__).resolve().parents[1] / "scripts/build_pilot_page.py")
_bp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_bp)

BANNED = tuple(dict.fromkeys(_bp.BANNED + ("understanding", "understand", "understands", "comprehension", "comprehend", "diagnosis",
                                           "diagnostic", "before anyone looked", "detecting", "unaware", "unconscious")))
_RX = re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(BANNED, key=len, reverse=True)) + r")\b", re.I)
TEMPLATE = "This line was replaced: it used wording StillHere does not use."
DISCLAIMER = "Not a diagnosis or medical advice."
REASONS = _bp.REASONS                                       # decline codes in plain words, the same map the static page uses


def hits(text: str) -> list:
    return [m.group(1).lower() for m in _RX.finditer(text.replace(DISCLAIMER, ""))]


def clean(text, template: str = TEMPLATE) -> str:
    """The text, or the template line if it holds a banned word or an em dash."""
    text = "" if text is None else str(text)
    return template if hits(text) or chr(0x2014) in text else text


def plain_reason(code: str) -> str:
    return REASONS.get(code, str(code).replace("_", " "))
