"""What the laptop-only question blocklist matches. One owner for the blocklist builder, egress.PublicQuestion and the public hygiene
scan, so all three agree. A text's terms are its lowercased words of 3+ letters and every run of PHRASE consecutive words; the
blocklist file is a 'salt <hex>' line, then one salted SHA-256 per blocked term (no word is stored). A rare word or a quoted phrase
from the clip is caught; an ordinary word on its own is not. Standard library only: the pre-push hook runs without the venv."""
import hashlib
import re
from pathlib import Path

PHRASE = 4


def words(text: str) -> list:
    return re.findall(r"[a-z']+", text.lower())


def terms(text: str) -> set:
    ws = words(text)
    return {w for w in ws if len(w) >= 3} | {" ".join(ws[i:i + PHRASE]) for i in range(len(ws) - PHRASE + 1)}


def term_hash(salt: bytes, term: str) -> str:
    return hashlib.sha256(salt + term.encode()).hexdigest()


def hits(text: str, salt: bytes, blocked: frozenset) -> bool:
    return any(term_hash(salt, t) in blocked for t in terms(text))


def load(path: Path) -> tuple:
    """(salt, hashes), or ValueError when the file is missing or malformed; callers fail closed on it."""
    if not path.is_file():
        raise ValueError(f"blocklist missing: {path}")
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    head = lines[0].split() if lines else []
    if len(head) != 2 or head[0] != "salt" or not re.fullmatch(r"[0-9a-f]{64}", head[1]) \
            or not all(re.fullmatch(r"[0-9a-f]{64}", h) for h in lines[1:]):
        raise ValueError(f"blocklist malformed: {path}")
    return bytes.fromhex(head[1]), frozenset(lines[1:])
