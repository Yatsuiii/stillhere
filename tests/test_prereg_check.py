"""Unit tests for the PREREG guard on synthetic files only (a fake PREREG, fake analysis code, a fake registry). No real analysis files."""
import hashlib
import json
from datetime import datetime, timezone

import pytest

from stillhere import prereg_check as G

V1 = "# Fake PREREG\n\n## 1. Data\n\ntext\n\n## 6. Statistics and bars\n\n| Bar | Pass if |\n|---|---|\n| **P1 (co-primary)** | a |\n| P2a | b |\n\n## Deviations\n\n"
V2 = V1 + "- **D1 Clarification.** one\n- **D10 Clarification.** ten\n"
V3 = V2 + "- **D22 Deviation.** parts\n  **Part B, wall wording.** text\n"
BEFORE, AFTER = datetime(2026, 10, 28, 12, tzinfo=timezone.utc), datetime(2026, 10, 29, 5, tzinfo=timezone.utc)


def sha(b):
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """Analysis root with PREREG (three appended versions), two registered entry points and a packaged copy; returns a registry writer."""
    monkeypatch.delenv("STILLHERE_PREREG_MODE", raising=False)
    monkeypatch.delenv("STILLHERE_ANALYSIS_ROOT", raising=False)
    root = tmp_path / "analysis"
    (root / "d22/gate1").mkdir(parents=True)
    (root / "d22/kds").mkdir()
    (root / "PREREG.md").write_text(V3)
    (root / "d22/gate1/gate1_bars.py").write_text("def p1(): pass\n")
    (root / "d22/gate1/stillhere_gate1.py").write_text("import gate1_bars\n")
    (root / "d22/kds/gate1_bars.py").write_text("def p1(): pass\n")              # byte copy packaged for Kaggle
    (root / "d22/gate1/test_gate1_bars.py").write_text("def test(): pass\n")      # tests are not entry points
    reg = {
        "analysis_root": "analysis", "hard_fail_from": "2026-10-29T00:00:00-04:00",
        "prereg": {"path": "PREREG.md", "chain": [{"bytes": len(v.encode()), "sha256": sha(v.encode())} for v in (V1, V2, V3)]},
        "entry_point_rule": {"roots": ["d22"], "patterns": ["stillhere_*.py", "*_bars.py", "*_scores.py"], "exclude": ["test_*.py"]},
        "entries": [
            {"path": "d22/gate1/gate1_bars.py", "sha256": sha(b"def p1(): pass\n"), "implements": ["§6", "P1", "P2a", "D10", "D22 Part B"]},
            {"path": "d22/gate1/stillhere_gate1.py", "sha256": sha(b"import gate1_bars\n"), "implements": ["§1", "D1"]},
        ],
        "not_entry_points": [],
    }

    def write(edit=None):
        r = json.loads(json.dumps(reg))
        if edit:
            edit(r)
        p = tmp_path / "prereg_registry.json"
        p.write_text(json.dumps(r))
        return p
    return root, write


def test_clean_tree_passes_even_in_hard_mode(tree):
    root, write = tree
    reg = write()
    assert G.check(reg) == []
    assert G.main(["--registry", str(reg)], now=AFTER) == 0


def test_changed_entry_point_is_found(tree):
    root, write = tree
    (root / "d22/gate1/gate1_bars.py").write_text("def p1(): return 'two-sided'\n")
    f = G.check(write())
    assert any(x.startswith("d22/gate1/gate1_bars.py changed") for x in f), f


def test_missing_entry_point_is_found(tree):
    root, write = tree
    (root / "d22/gate1/stillhere_gate1.py").unlink()
    assert "registered file missing: d22/gate1/stillhere_gate1.py" in G.check(write())


@pytest.mark.parametrize("bad, why", [("D99", "D99 is not a PREREG entry"), ("P9", "P9 is not a PREREG bar"),
                                      ("§12", "§12 is not a PREREG section"), ("D22 Part Z", "'Part Z' does not appear in D22"),
                                      ("bar one", "malformed PREREG ID")])
def test_unknown_or_malformed_id_is_found(tree, bad, why):
    root, write = tree
    f = G.check(write(lambda r: r["entries"][1]["implements"].append(bad)))
    assert len(f) == 1 and why in f[0], f


def test_d1_is_not_satisfied_by_d10_alone():
    secs, bars, entries = G.prereg_ids(V1 + "- **D10 Clarification.** ten\n")
    assert G.id_problem("D10", secs, bars, entries) is None
    assert G.id_problem("D1", secs, bars, entries) == "D1 is not a PREREG entry"


def test_entry_without_ids_is_found(tree):
    root, write = tree
    f = G.check(write(lambda r: r["entries"][1].update(implements=[])))
    assert f == ["d22/gate1/stillhere_gate1.py cites no PREREG ID"]


def test_unregistered_entry_point_is_found_and_exemptions_hold(tree):
    root, write = tree
    (root / "d22/gate1/new_scores.py").write_text("def z(): pass\n")
    (root / "d22/gate1/stillhere_tpl_kernel.py").write_text("# template\n")
    f = G.check(write(lambda r: r["not_entry_points"].append({"path": "d22/gate1/stillhere_tpl_kernel.py", "reason": "template"})))
    assert f == ["unregistered analysis entry point: d22/gate1/new_scores.py (register it with its PREREG IDs, or list it under not_entry_points)"]


def test_modified_packaged_copy_is_flagged(tree):
    root, write = tree
    (root / "d22/kds/gate1_bars.py").write_text("def p1(): return 1\n")
    f = G.check(write())
    assert len(f) == 1 and "d22/kds/gate1_bars.py" in f[0], f


def test_prereg_append_without_new_chain_head_is_found(tree):
    root, write = tree
    (root / "PREREG.md").write_text(V3 + "- **D24 Deviation.** new\n")
    f = G.check(write())
    assert len(f) == 1 and f[0].startswith("PREREG is not the registered chain head"), f


def test_prereg_rewrite_breaks_the_chain(tree):
    root, write = tree
    (root / "PREREG.md").write_text(V3.replace("| P2a | b |", "| P2a | c |"))
    f = G.check(write())
    assert any("not the registered chain head" in x for x in f)
    assert sum("is not an append of chain version" in x for x in f) == 2, f        # V1 and V2 prefixes both broken


def test_missing_analysis_root_is_a_finding(tree, monkeypatch, tmp_path):
    root, write = tree
    monkeypatch.setenv("STILLHERE_ANALYSIS_ROOT", str(tmp_path / "nowhere"))
    f = G.check(write())
    assert len(f) == 1 and f[0].startswith("analysis root not found"), f


def test_warn_before_the_date_hard_after(tree, capsys):
    root, write = tree
    (root / "d22/gate1/new_scores.py").write_text("def z(): pass\n")
    reg = write()
    assert G.main(["--registry", str(reg)], now=BEFORE) == 0
    assert "  WARN unregistered analysis entry point" in capsys.readouterr().out
    assert G.main(["--registry", str(reg)], now=AFTER) == 1
    assert "  FAIL unregistered analysis entry point" in capsys.readouterr().out


def test_hard_fail_starts_at_midnight_eastern(tree):
    root, write = tree
    reg = write()
    assert G.mode(reg, datetime(2026, 10, 29, 3, 59, tzinfo=timezone.utc)) == "warn"
    assert G.mode(reg, datetime(2026, 10, 29, 4, 0, tzinfo=timezone.utc)) == "hard"


def test_env_overrides_the_date(tree, monkeypatch):
    root, write = tree
    (root / "d22/gate1/new_scores.py").write_text("def z(): pass\n")
    reg = write()
    monkeypatch.setenv("STILLHERE_PREREG_MODE", "hard")
    assert G.main(["--registry", str(reg)], now=BEFORE) == 1
    monkeypatch.setenv("STILLHERE_PREREG_MODE", "off")
    with pytest.raises(ValueError):
        G.mode(reg, BEFORE)


def test_repo_scope_checks_only_this_repo_and_never_weakens_a_full_tree(tmp_path, monkeypatch):
    repo = tmp_path / "stillhere-app"; (repo / "stillhere").mkdir(parents=True)
    f = repo / "stillhere" / "heldout.py"; f.write_text("x = 1\n")
    reg = {"analysis_root": "../stillhere", "hard_fail_from": "2000-01-01T00:00:00+00:00", "prereg": {"path": "PREREG.md", "chain": []},
           "entry_point_rule": {"roots": ["d22"], "patterns": ["*.py"]}, "not_entry_points": [],
           "entries": [{"path": "../stillhere-app/stillhere/heldout.py", "sha256": hashlib.sha256(b"x = 1\n").hexdigest(), "implements": ["§8"]},
                       {"path": "d22/elsewhere.py", "sha256": "0" * 64, "implements": ["D1"]}]}
    r = repo / "prereg_registry.json"; r.write_text(json.dumps(reg))
    monkeypatch.delenv("STILLHERE_ANALYSIS_ROOT", raising=False); monkeypatch.delenv("STILLHERE_PREREG_MODE", raising=False)
    assert G.check(r) == ["analysis root not found: " + str((tmp_path / "stillhere").resolve()) + " (set STILLHERE_ANALYSIS_ROOT)"]
    monkeypatch.setenv("STILLHERE_GUARD_SCOPE", "repo")
    assert G.repo_scope(r) and G.check(r) == [] and G.main(["--registry", str(r)]) == 0          # hard mode, still clean
    f.write_text("x = 2\n")
    assert len(G.check(r)) == 1 and "changed" in G.check(r)[0] and G.main(["--registry", str(r)]) == 1
    (tmp_path / "stillhere").mkdir()                                                          # a full tree is present: scope ignored
    assert not G.repo_scope(r) and any("PREREG not found" in x for x in G.check(r))


def test_repo_copy_of_prereg_must_be_the_chain_head(tree, tmp_path, monkeypatch):
    """Check 5: a PREREG.md next to the registry is verified in full and in repo scope; an older or edited copy is a finding."""
    root, write = tree
    reg = write()
    (tmp_path / "PREREG.md").write_text(V3)
    assert G.check(reg) == []
    (tmp_path / "PREREG.md").write_text(V2)
    assert [x for x in G.check(reg) if x.startswith("repo copy PREREG is not the registered chain head")]
    monkeypatch.setenv("STILLHERE_ANALYSIS_ROOT", str(tmp_path / "nowhere")); monkeypatch.setenv("STILLHERE_GUARD_SCOPE", "repo")
    assert G.repo_scope(reg) and any(x.startswith("repo copy PREREG") for x in G.check(reg))


def test_this_repos_prereg_is_the_frozen_d0_d27_file():
    """The PREREG.md shipped in this repository is the frozen D0-D27 file, byte for byte (ruling of Oct 29, 17:35 UTC, item 11)."""
    reg = json.loads(G.REGISTRY.read_text()); head = reg["prereg"]["chain"][-1]
    data = (G.REGISTRY.parent / "PREREG.md").read_bytes()
    assert (len(data), sha(data)) == (head["bytes"], head["sha256"]) == (98295, "310854864a0154ebeb1b31201010192f8a4c577c73d61883078d20acd78a3c36")
    assert G.check_repo_copy(G.REGISTRY, reg["prereg"]["chain"]) == []
