"""scripts/paranoia_files.py (what scripts/reproduce_paranoia.sh fetches and scripts/make_paranoia_release.sh stages; cloud review of Nov 2,
12:15 UTC, item 6), offline: every expected SHA-256 comes from the committed scorer, runner and our scoring record; a fetched file with
other bytes is refused and removed; tribe_common is written only from the R2 kernel's COMMON with its pinned hash; only our pinned word
tables seed the forecasts; only our pinned files are staged for the release."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paranoia_files", REPO / "scripts/paranoia_files.py")
PF = importlib.util.module_from_spec(spec); spec.loader.exec_module(PF)


def test_pins_are_the_committed_ones():
    p = PF.pins()
    src = (REPO / "scripts/r2_paranoia.py").read_text()
    for name, h in p.items():
        if name.endswith((".wav", ".tsv", ".npz", "surprisal.json")):
            assert h in src, name                                                   # restated nowhere: read from the scorer
    read = json.loads((REPO / "results/r2_paranoia_SHA256.json").read_text())
    assert all(p[a] == read[a] for a in PF.ATLAS) and len([n for n in p if n.startswith("pred_stock_")]) == 9
    assert p["story_fulltext.txt"] in (REPO / "r2/run_r2_forecast.py").read_text()


def test_fetch_refuses_other_bytes_and_removes_them(tmp_path):
    src = tmp_path / "src.bin"; src.write_bytes(b"the right bytes")
    want = hashlib.sha256(b"the right bytes").hexdigest()
    PF.fetch(src.as_uri(), str(tmp_path / "d/x.bin"), want)
    assert (tmp_path / "d/x.bin").read_bytes() == b"the right bytes"
    src.write_bytes(b"other bytes"); (tmp_path / "d/x.bin").unlink()
    with pytest.raises(SystemExit, match="refused: x.bin"):
        PF.fetch(src.as_uri(), str(tmp_path / "d/x.bin"), want)
    assert not (tmp_path / "d/x.bin").exists()


def test_common_is_the_kernels_with_its_pinned_hash(tmp_path, monkeypatch):
    PF.common(str(tmp_path))
    c = (tmp_path / "k2/tribe_common.py").read_text()
    assert hashlib.sha256(c.encode()).hexdigest() == PF.COMMON_SHA
    assert not (REPO / "k2").exists()                                               # nothing copied into the repository
    monkeypatch.setattr(PF, "COMMON_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="COMMON has SHA-256"):
        PF.common(str(tmp_path / "other"))


def test_only_our_word_tables_seed_the_forecasts(tmp_path):
    for s in PF.STORIES:
        (tmp_path / f"words_{s}.tsv").write_text("text\tstart\tduration\nw\t1.0\t0.2\n")
    with pytest.raises(SystemExit, match="is not our word table"):
        PF.seed_words(str(tmp_path / "data"), str(tmp_path))
    assert not (tmp_path / "data/r2/forecast/tmp/story1_full/stim.tsv").exists()


def test_only_our_pinned_files_are_staged(tmp_path):
    for s in PF.STORIES:
        (tmp_path / f"words_{s}.tsv").write_text("w")
        for k in PF.MODES:
            (tmp_path / f"pred_stock_{s}_{k}.npz").write_bytes(b"not ours")
    with pytest.raises(SystemExit, match="not our pinned files"):
        PF.stage_release(str(tmp_path), str(tmp_path / "stage"), None, False)
    assert not (tmp_path / "stage").exists()
    assert "CC BY-NC 4.0" in PF.README and "d'Ascoli et al. 2026" in PF.README and "ds001338" in PF.README


def test_a_release_that_answers_404_says_why(tmp_path, monkeypatch):
    import urllib.error
    def missing(url, dst, *a, **k):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
    monkeypatch.setattr(PF.netfetch, "get", missing)
    with pytest.raises(SystemExit, match="answered 404. The release paranoia-forecasts-v1 is public once the repository is"):
        PF.fetch_release(str(tmp_path / "rel"), forecasts=False)
