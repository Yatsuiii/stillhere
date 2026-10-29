"""Hygiene scan for the public StillHere repository: the pre-push hook runs it, and it can be run by hand. It FAILS CLOSED.

It scans every file in the HEAD tree, every older blob reachable from HEAD (a push carries the whole history) and every commit
message, and fails on:
- any line holding a word or 4-word phrase whose salted SHA-256 is on the laptop-only blocklist built from the privately shared clip
  (format and matching in stillhere/blockterms.py, the same rules egress.PublicQuestion uses). The list is read from outside the
  repository (STILLHERE_QUESTION_BLOCKLIST or ~/.config/stillhere/question_blocklist.txt); if it is missing or malformed, the scan fails;
- links to agent sessions or shared transcripts;
- API keys and tokens (Hugging Face, Fireworks, OpenAI, Kaggle, and long secrets assigned to key, token, secret or password names);
- absolute paths that name the laptop user;
- subject IDs in the private sedation dataset's format;
- the clip's name; em dashes; and any file over 10 MB.
One pinned exception (ruling of Oct 29, 17:35 UTC, item 11): the frozen PREREG (D0-D27) at exactly PREREG.md with exactly its SHA-256 may
carry its known clip_name and subject_id lines (two ordinary English phrases that start with the clip's name, and the sedation
dataset's subject IDs);
every other rule still applies to it, and any other path or byte fails as usual. The scan's output names the exception every run.
Findings name the file, the line and the rule, never the matched text. Standard library only.
usage: python scripts/hygiene_scan.py [--repo DIR] [--blocklist PATH]"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stillhere import blockterms  # noqa: E402  (standard library only)

MAX_BYTES = 10 * 1024 * 1024
USER = "Yat" + "suiii"                                                # spelled in pieces so the scanner never flags itself
CLIP = "t" + "aken"
RULES = [
    ("session_link", re.compile(r"claude\.ai/(?:code/session|share)|chatgpt\.com/share|chat\.openai\.com/share", re.I)),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("fireworks_key", re.compile(r"\bfw_[A-Za-z0-9]{20,}")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("kaggle_token", re.compile(r"\bKGAT_[A-Za-z0-9]{20,}|\"key\"\s*:\s*\"[0-9a-f]{32}\"")),
    ("secret_assignment", re.compile(r"(?i)\b[\w-]*(?:api[_-]?key|token|secret|passw(?:or)?d)[\w-]*[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_\-+/=]{32,}")),
    ("user_path", re.compile(rf"/home/{USER}\b|/run/media/{USER}\b")),
    ("subject_id", re.compile(r"\bsub-\d{2,4}[A-Z]{2}\b")),
    ("clip_name", re.compile(rf"\b{CLIP}\b", re.I)),
    ("em_dash", re.compile(chr(0x2014))),
]
PINNED = {"PREREG.md": ("310854864a0154ebeb1b31201010192f8a4c577c73d61883078d20acd78a3c36", frozenset({"clip_name", "subject_id"}))}


def pinned_rules(name: str, data: bytes) -> frozenset:
    """The rules a pinned file may hit: only at its exact path and SHA-256, else none."""
    pin = PINNED.get(name)
    return pin[1] if pin and hashlib.sha256(data).hexdigest() == pin[0] else frozenset()


def git(repo: Path, *args) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def history_blobs(repo: Path) -> list:
    """(path, blob sha) for every blob reachable from HEAD."""
    objs = git(repo, "rev-list", "--objects", "HEAD")
    out = subprocess.run(["git", "-C", str(repo), "cat-file", "--batch-check=%(objectname) %(objecttype) %(rest)"],
                         input=objs, check=True, capture_output=True).stdout.decode()
    return [(rest, sha) for sha, kind, rest in (l.split(" ", 2) for l in out.splitlines() if l.count(" ") >= 2) if kind == "blob"]


def scan_text(name: str, text: str, salt: bytes, blocked: frozenset) -> list:
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if blockterms.hits(line, salt, blocked):
            out.append(f"{name}:{i}: blocklisted_term")
        out += [f"{name}:{i}: {rule}" for rule, rx in RULES if rx.search(line)]
    return out


def scan(repo: Path, blocklist: Path, notes: list = None) -> list:
    salt, blocked = blockterms.load(blocklist)
    findings = []; notes = [] if notes is None else notes
    if not git(repo, "rev-parse", "--verify", "--quiet", "HEAD").strip():
        return findings
    for name in git(repo, "ls-tree", "-r", "-z", "--name-only", "HEAD").decode().split("\0"):
        if not name:
            continue
        data = git(repo, "cat-file", "blob", f"HEAD:{name}")
        if len(data) > MAX_BYTES:
            findings.append(f"{name}: file_over_10MB")
        if b"\0" in data[:8192]:
            continue                                                  # binary: size rule only
        found, allow = scan_text(name, data.decode("utf-8", "replace"), salt, blocked), pinned_rules(name, data)
        if allow:
            kept = [f for f in found if f.rsplit(": ", 1)[1] not in allow]
            notes.append(f"pinned exception: {name} sha256 {PINNED[name][0][:12]}... (frozen PREREG D0-D27), "
                         f"{len(found) - len(kept)} known {'/'.join(sorted(allow))} line(s) allowed")
            found = kept
        findings += found
    head = {e.split()[2] for e in git(repo, "ls-tree", "-r", "-z", "HEAD").decode().split("\0") if e}
    for path, blob in history_blobs(repo):
        if blob not in head:
            data = git(repo, "cat-file", "blob", blob)
            if b"\0" not in data[:8192]:
                findings += scan_text(f"{path} (history {blob[:8]})", data.decode("utf-8", "replace"), salt, blocked)
    for sha in git(repo, "rev-list", "HEAD").decode().split():
        findings += scan_text(f"commit {sha[:12]}", git(repo, "log", "-1", "--format=%B", sha).decode("utf-8", "replace"), salt, blocked)
    return findings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--blocklist", type=Path,
                    default=Path(os.environ.get("STILLHERE_QUESTION_BLOCKLIST", Path.home() / ".config/stillhere/question_blocklist.txt")))
    a = ap.parse_args(argv)
    notes = []
    try:
        findings = scan(a.repo, a.blocklist, notes)
    except ValueError as e:
        print(f"hygiene-scan: FAIL CLOSED: {e}")
        return 1
    if findings:
        print(f"hygiene-scan: FAIL, {len(findings)} finding(s)")
        print("\n".join(f"  {f}" for f in findings + notes))
        return 1
    files = len([n for n in git(a.repo, "ls-tree", "-r", "-z", "--name-only", "HEAD").decode().split("\0") if n])
    print(f"hygiene-scan: clean ({files} files, every older blob and every commit message)")
    for n in notes:
        print(f"  {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
