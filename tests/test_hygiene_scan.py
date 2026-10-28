"""The hygiene scan on throwaway git repositories with a temporary blocklist: a planted blocklisted word fails, a clean tree passes, a
missing or malformed blocklist fails closed, and each other rule (links, keys, user paths, subject IDs, the clip name, em dashes,
size, commit messages) fails when planted."""
import hashlib
import importlib.util
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("hygiene", Path(__file__).resolve().parents[1] / "scripts" / "hygiene_scan.py")
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
SALT = bytes(range(32))


def blocklist(tmp_path, words=("zorblax",)):
    p = tmp_path / "bl.txt"
    p.write_text(f"# test\nsalt {SALT.hex()}\n" + "".join(hashlib.sha256(SALT + w.encode()).hexdigest() + "\n" for w in words))
    return p


def repo(tmp_path, files, message="clean commit"):
    r = tmp_path / "r"; r.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(r), *a], check=True, capture_output=True)
    run("init", "-q"); run("config", "user.name", "t"); run("config", "user.email", "t@example.org")
    for name, data in files.items():
        f = r / name; f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data if isinstance(data, bytes) else data.encode())
    run("add", "-A"); run("commit", "-q", "-m", message)
    return r


def test_clean_tree_passes_and_planted_word_fails(tmp_path):
    bl = blocklist(tmp_path)
    assert H.main(["--repo", str(repo(tmp_path, {"a.py": "x = 1\n# a normal comment\n"})), "--blocklist", str(bl)]) == 0
    r2 = tmp_path / "two"; r2.mkdir()
    assert H.scan(repo(r2, {"notes.md": "line one\nthe Zorblax returns\n"}), bl) == ["notes.md:2: blocklisted_word"]


def test_missing_or_malformed_blocklist_fails_closed(tmp_path):
    r = repo(tmp_path, {"a.py": "x = 1\n"})
    assert H.main(["--repo", str(r), "--blocklist", str(tmp_path / "missing.txt")]) == 1
    bad = tmp_path / "bad.txt"; bad.write_text("zorblax\n")                        # plaintext words are malformed
    assert H.main(["--repo", str(r), "--blocklist", str(bad)]) == 1


@pytest.mark.parametrize("text,rule", [
    ("see https://claude" + ".ai/code/session_" + "abc", "session_link"),
    ("token " + "hf_" + "A" * 34, "hf_token"),
    ("fw_" + "B" * 24, "fireworks_key"),
    ("api_key = '" + "c" * 40 + "'", "secret_assignment"),
    ("/home/" + H.USER + "/x", "user_path"),
    ("subject " + "sub-" + "22TK", "subject_id"),
    ("the clip " + H.CLIP.capitalize(), "clip_name"),
    ("a" + chr(0x2014) + "b", "em_dash"),
])
def test_each_rule_fails_when_planted(tmp_path, text, rule):
    findings = H.scan(repo(tmp_path, {"f.txt": text + "\n"}), blocklist(tmp_path))
    assert any(f.endswith(rule) for f in findings), findings


def test_commit_messages_and_size_are_checked(tmp_path, monkeypatch):
    r = repo(tmp_path, {"a.txt": "fine\n"}, message="see https://chatgpt" + ".com/share/x")
    assert any(f.startswith("commit ") and f.endswith("session_link") for f in H.scan(r, blocklist(tmp_path)))
    monkeypatch.setattr(H, "MAX_BYTES", 10)
    r2 = tmp_path / "two"; r2.mkdir()
    assert any(f.endswith("file_over_10MB") for f in H.scan(repo(r2, {"big.bin": b"\0" * 64}), blocklist(tmp_path)))
