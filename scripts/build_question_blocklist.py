"""Builds the laptop-only question blocklist (egress.PublicQuestion) from the privately shared clip. Rules (cloud thread, Oct 28): Yatsu
decrypts the clip into their own home folder; this script transcribes it with a LOCAL faster-whisper model only (never Voice Transcribe
or any API, offline) and keeps the transcript in memory only. It blocks (terms as in stillhere/blockterms.py) every run of 4 words, and
every word of 3+ letters that is uncommon or a name: uncommon means outside stillhere/common_words.txt and wordfreq's 25k most frequent
English words (a laptop-only build dependency), and a name is a word capitalised after the first word of a sentence, however frequent.
It writes only salted SHA-256s to ~/.config/stillhere/question_blocklist.txt (mode 600, no word stored), and prints only the
blocklist file's SHA-256. It deletes the decrypted audio whatever happens unless --keep-audio is given (Yatsu keeps the decrypted
clip in their own home folder until the project ends, Nov 9). Nothing here is committed, logged or sent anywhere.
usage: HF_HUB_OFFLINE=1 PYTHONPATH=.:<faster-whisper and wordfreq site-packages> python scripts/build_question_blocklist.py AUDIO [--keep-audio]"""
import hashlib
import os
import re
import sys
from pathlib import Path

from stillhere import blockterms, egress

REPO = Path(__file__).resolve().parents[1]
COMMON = REPO / "stillhere" / "common_words.txt"
MODEL = "Systran/faster-whisper-base.en"
FREQ_TOP_N = 25000


def transcribe(audio: Path) -> str:
    """Local faster-whisper on the CPU, cached model files only; the text stays in memory."""
    from faster_whisper import WhisperModel
    m = WhisperModel(MODEL, device="cpu", compute_type="int8", local_files_only=True)
    segs, _ = m.transcribe(str(audio), language="en")
    return "\n".join(s.text.strip() for s in segs)                  # one segment per line: a segment starts a sentence


def common_words() -> set:
    from wordfreq import top_n_list
    listed = {l.strip() for l in COMMON.read_text().splitlines() if l.strip() and not l.startswith("#")}
    return listed | set(top_n_list("en", FREQ_TOP_N))


def names(text: str) -> set:
    """Words capitalised after the first word of a sentence or line and never written in lowercase (I, I'm and the like excepted)."""
    lower = set(re.findall(r"\b[a-z][a-z']*", text))
    return {w.lower() for sentence in re.split(r"[.!?]+\s+|\n", text) for w in re.findall(r"[A-Za-z']+", sentence)[1:]
            if w[0].isupper() and w.lower() not in lower and w != "I" and not w.startswith("I'")}


def blocked_terms(text: str, common: set) -> set:
    rare = names(text) | {w for w in blockterms.words(text) if w not in common}
    return {t for t in blockterms.terms(text) if " " in t or t in rare}


def build(audio: Path, out: Path = egress.BLOCKLIST, transcriber=transcribe, keep_audio: bool = False, common=common_words) -> str:
    audio = audio.resolve()
    if Path.home().resolve() not in audio.parents or REPO in audio.parents:
        raise SystemExit("refused: the decrypted audio must sit in Yatsu's own home folder, outside the repository")
    try:
        blocked = blocked_terms(transcriber(audio), common())
    finally:
        if not keep_audio:
            audio.unlink(missing_ok=True)                              # default: delete the decrypted audio right after, whatever happens
    salt = os.urandom(32)
    out.parent.mkdir(parents=True, exist_ok=True); os.chmod(out.parent, 0o700)
    body = "# stillhere question blocklist: salted SHA-256 of blocked words and phrases (laptop only; never committed, logged or sent)\n" \
           + f"salt {salt.hex()}\n" + "".join(blockterms.term_hash(salt, t) + "\n" for t in sorted(blocked))
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(body)
    os.chmod(out, 0o600)
    return hashlib.sha256(out.read_bytes()).hexdigest()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--keep-audio"]
    print(build(Path(args[0]), keep_audio="--keep-audio" in sys.argv[1:]))
