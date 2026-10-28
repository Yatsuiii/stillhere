"""Card verifier for the analysis desk (PREREG D18 + D23, PLAN §6). A card is shown only if `verify(card)` returns no violations.

Fixed lines (allowed verbatim, exempt from the word blocklist): the research header, "Not a diagnosis or medical advice." and the
clinical_out_of_scope decline. Everything else on a card is scanned for the blocked words. Verdict words are for group results only:
a card about one volunteer may carry only its z and the label "group-level tool"."""
import re

HEADER = "Research readout. Healthy volunteers, public data. Same-clip controls scanned: {n}. Not a diagnosis or medical advice."
DISCLAIMER = "Not a diagnosis or medical advice."
DECLINE_CLINICAL = ("StillHere answers pre-registered research questions on healthy-volunteer data. "
                    "It gives no medical advice and says nothing about any patient.")
RESPONSE_SHOWN = "response shown (research readout)"
NOT_SHOWN = "not shown (inconclusive)"
GROUP_LEVEL = "group-level tool"

# D23 verifier list (outside the fixed lines), D18 awareness ban, and D23's never-list phrases
_WORDS = [r"follow-?up", r"worth a", r"preserved", r"recommend\w*", r"should", r"refer\w*", r"patients?", r"hospitals?", r"clinics?",
          r"clinical", r"doctors?", r"diagnos\w*", r"prognos\w*", r"treatment\w*", r"life support", r"triage", r"aware\w*", r"conscious\w*",
          r"marks patients", r"any story", r"any language"]
_BLOCK = re.compile(r"\b(" + "|".join(_WORDS) + r")\b", re.IGNORECASE)


def _strip_fixed(text: str) -> str:
    text = re.sub(re.escape(HEADER).replace(re.escape("{n}"), r"\d+"), " ", text)
    return text.replace(DECLINE_CLINICAL, " ").replace(DISCLAIMER, " ")


def blocked_words(text: str) -> list[str]:
    """The blocked words in `text` outside the fixed lines (also used on model text that is never shown)."""
    return [m.group(0) for m in _BLOCK.finditer(_strip_fixed(text))]


def verify(card: dict) -> list[str]:
    """card: {"subject": {"kind": "group", "n": int} | {"kind": "volunteer", "id": str}, "text": str, "controls_scanned": int}."""
    out = []
    text, subj = card["text"], card["subject"]
    header = HEADER.format(n=card["controls_scanned"])
    if not text.startswith(header):
        out.append("missing the fixed research header")
    out += [f"blocked word: {w!r}" for w in blocked_words(text)]
    verdict = RESPONSE_SHOWN in text or NOT_SHOWN in text
    if subj["kind"] == "volunteer":
        if verdict:
            out.append("verdict words on a single-volunteer card")
        if GROUP_LEVEL not in text:
            out.append("single-volunteer card without 'group-level tool'")
    elif subj["kind"] == "group":
        if verdict and DISCLAIMER not in text:
            out.append("group verdict without 'Not a diagnosis or medical advice.'")
    else:
        out.append("subject must be a group or a volunteer")
    return out
