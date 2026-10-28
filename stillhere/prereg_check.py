"""PREREG guard (SPARK-PLAN D): fails when the analysis code and the frozen pre-registration drift apart.

The registry (prereg_registry.json) lists every analysis entry point with its SHA-256 and the PREREG IDs it implements. The tags live
in the registry, not in the frozen files, so registering never edits analysis code. Checks:
1. PREREG.md is byte-identical to the registry's chain head, and every earlier chain version is a byte prefix of it (append only).
2. Every registered file exists and matches its SHA-256.
3. Every cited ID exists in the PREREG text. ID forms: "§N" needs a "## N." heading; "Pn" or "Pn<letter>" needs a bar-table row
   starting "| Pn" or "| **Pn"; "Dn" needs an entry line "- **Dn" (D1 is not satisfied by D10); "Dn <text>" also needs <text> inside
   that entry, continuation lines included (e.g. "D22 Part B").
4. Every file under the scan roots whose name matches an entry-point pattern is registered, is a byte copy of a registered file (a
   packaged Kaggle input), or is listed under not_entry_points with a reason.
Before hard_fail_from (2026-10-29 00:00 ET) findings are warnings and the exit code is 0; from then on any finding exits 1.
STILLHERE_PREREG_MODE=warn|hard overrides the date; STILLHERE_ANALYSIS_ROOT overrides the registry's analysis root. Stdlib only, so
the git and Muse Code hooks can run it without the project venv.

usage: python -m stillhere.prereg_check [--registry PATH]"""
import argparse
import fnmatch
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent.parent / "prereg_registry.json"
_ID = re.compile(r"^(?:§(?P<sec>\d+)|(?P<bar>P\d+[a-z]?)|(?P<dec>D\d+)(?: (?P<qual>.+))?)$")
_ENTRY = re.compile(r"^\s*- \*\*(D\d+)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prereg_ids(text: str) -> tuple[set, set, dict]:
    """Section numbers, bar names and D-entries (ID -> the entry's text up to the next entry or heading) in the PREREG text."""
    secs = set(re.findall(r"^## (\d+)\. ", text, re.M))
    bars = set(re.findall(r"^\| \**(P\d+[a-z]?)\b", text, re.M))
    entries, cur = {}, None
    for line in text.splitlines():
        m = _ENTRY.match(line)
        if m:
            cur = m.group(1)
        elif line.startswith("#"):
            cur = None
        if cur:
            entries[cur] = entries.get(cur, "") + line + "\n"
    return secs, bars, entries


def id_problem(i: str, secs: set, bars: set, entries: dict):
    m = _ID.match(i)
    if not m:
        return f"malformed PREREG ID {i!r} (use §N, Pn, Dn or 'Dn <text>')"
    if m["sec"] and m["sec"] not in secs:
        return f"§{m['sec']} is not a PREREG section"
    if m["bar"] and m["bar"] not in bars:
        return f"{m['bar']} is not a PREREG bar"
    if m["dec"]:
        if m["dec"] not in entries:
            return f"{m['dec']} is not a PREREG entry"
        if m["qual"] and m["qual"] not in entries[m["dec"]]:
            return f"{i!r}: {m['qual']!r} does not appear in {m['dec']}"
    return None


def check_chain(data: bytes, chain: list) -> list:
    head, out = chain[-1], []
    if len(data) != head["bytes"] or hashlib.sha256(data).hexdigest() != head["sha256"]:
        out.append(f"PREREG is not the registered chain head {head['sha256'][:12]}... ({head['bytes']} B): it is "
                   f"{hashlib.sha256(data).hexdigest()[:12]}... ({len(data)} B). Add the new version to the chain once its hash is posted.")
    for v in chain[:-1]:
        if hashlib.sha256(data[:v["bytes"]]).hexdigest() != v["sha256"]:
            out.append(f"PREREG is not an append of chain version {v['sha256'][:12]}...: its first {v['bytes']} B changed")
    return out


def unregistered(root: Path, rule: dict, registered: dict, skip: set) -> list:
    copies, out = set(registered.values()), []
    for top in rule["roots"]:
        for dirpath, dirnames, files in os.walk(root / top):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__pycache__")
            for f in sorted(files):
                if not any(fnmatch.fnmatch(f, p) for p in rule["patterns"]) or any(fnmatch.fnmatch(f, p) for p in rule.get("exclude", [])):
                    continue
                p = Path(dirpath) / f
                rel = p.relative_to(root).as_posix()
                if rel not in registered and rel not in skip and sha256(p) not in copies:
                    out.append(f"unregistered analysis entry point: {rel} (register it with its PREREG IDs, or list it under not_entry_points)")
    return out


REPO_SCOPE_NOTE = "repo scope: the analysis tree and the PREREG chain are not on this machine by design; the laptop's pre-commit guard checks them"


def repo_scope(registry: Path) -> bool:
    """STILLHERE_GUARD_SCOPE=repo, for the Muse Code clone that never sees the analysis tree. Ignored wherever the analysis root
    exists, so it can never weaken the full check on the laptop."""
    reg = json.loads(registry.read_text())
    root = Path(os.environ.get("STILLHERE_ANALYSIS_ROOT") or (registry.parent / reg["analysis_root"])).resolve()
    return os.environ.get("STILLHERE_GUARD_SCOPE") == "repo" and not root.is_dir()


def check_repo(registry: Path) -> list:
    """Repo scope: the registry's own entries for this repository (paths ../<repo>/...) must match their registered SHA-256 and cite
    PREREG IDs; their ID validity needs PREREG and is checked on the laptop."""
    reg = json.loads(registry.read_text()); prefix = f"../{registry.parent.resolve().name}/"; out = []
    for e in reg["entries"]:
        if not e["path"].startswith(prefix):
            continue
        p = registry.parent / e["path"][len(prefix):]
        if not p.is_file():
            out.append(f"registered file missing: {e['path']}")
        elif sha256(p) != e["sha256"]:
            out.append(f"{e['path']} changed: {sha256(p)[:12]}... is not the registered {e['sha256'][:12]}...")
        if not e["implements"]:
            out.append(f"{e['path']} cites no PREREG ID")
    return out


def check(registry: Path = REGISTRY) -> list:
    """All findings, empty when the registry, the analysis files and PREREG agree."""
    if repo_scope(registry):
        return check_repo(registry)
    reg = json.loads(registry.read_text())
    root = Path(os.environ.get("STILLHERE_ANALYSIS_ROOT") or (registry.parent / reg["analysis_root"])).resolve()
    if not root.is_dir():
        return [f"analysis root not found: {root} (set STILLHERE_ANALYSIS_ROOT)"]
    prereg = root / reg["prereg"]["path"]
    if not prereg.is_file():
        return [f"PREREG not found: {prereg}"]
    data = prereg.read_bytes()
    out = check_chain(data, reg["prereg"]["chain"])
    ids = prereg_ids(data.decode("utf-8"))
    for e in reg["entries"]:
        p = root / e["path"]
        if not p.is_file():
            out.append(f"registered file missing: {e['path']}")
            continue
        got = sha256(p)
        if got != e["sha256"]:
            out.append(f"{e['path']} changed: {got[:12]}... is not the registered {e['sha256'][:12]}...")
        if not e["implements"]:
            out.append(f"{e['path']} cites no PREREG ID")
        out += [f"{e['path']}: {why}" for why in (id_problem(i, *ids) for i in e["implements"]) if why]
    skip = {s["path"] for s in reg.get("not_entry_points", [])}
    out += unregistered(root, reg["entry_point_rule"], {e["path"]: e["sha256"] for e in reg["entries"]}, skip)
    return out


def mode(registry: Path = REGISTRY, now: datetime = None) -> str:
    forced = os.environ.get("STILLHERE_PREREG_MODE")
    if forced:
        if forced not in ("warn", "hard"):
            raise ValueError(f"STILLHERE_PREREG_MODE must be warn or hard, not {forced!r}")
        return forced
    start = datetime.fromisoformat(json.loads(registry.read_text())["hard_fail_from"])
    return "hard" if (now or datetime.now(timezone.utc)) >= start else "warn"


def main(argv=None, now: datetime = None) -> int:
    ap = argparse.ArgumentParser(prog="prereg_check", description=__doc__.splitlines()[0])
    ap.add_argument("--registry", type=Path, default=REGISTRY)
    a = ap.parse_args(argv)
    m, findings = mode(a.registry, now), check(a.registry)
    if repo_scope(a.registry):
        print(f"prereg-guard: {REPO_SCOPE_NOTE}")
    if not findings:
        print(f"prereg-guard: clean ({m} mode)")
        return 0
    tag = "FAIL" if m == "hard" else "WARN"
    print(f"prereg-guard: {len(findings)} finding(s), {m} mode" + ("" if m == "hard" else " (exit 0 until hard_fail_from)"))
    for f in findings:
        print(f"  {tag} {f}")
    return 1 if m == "hard" else 0


if __name__ == "__main__":
    sys.exit(main())
