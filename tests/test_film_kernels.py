"""The three film forecast kernels (cloud rulings of Oct 31, 05:15 and 05:30 UTC): every module-level name they read is bound (audit K2:
film-rem and film-am wrote their record with an unbound `name`), the builder reads the template beside it and rebuilds the committed
kernels byte for byte (K5), all three carry one lock, one set of Hugging Face pins and tribe_common.stitched_predict byte for byte, and
the events stage builds words before chunking and predicts through stitched_predict, never tribev2's own events path or predict."""
import ast
import builtins
import hashlib
import json
import subprocess
import symtable
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FILM = REPO / "kernels" / "film"
KERNELS = {"intact": FILM / "intact/stillhere_pilot_bridge_intact.py", "rem": FILM / "rem/stillhere_pilot_film_rem.py",
           "am": FILM / "am/stillhere_pilot_film_am.py"}
COMMON = REPO.parent / "stillhere" / "k2" / "tribe_common.py"                # kept outside the repository
STIM_MANIFEST = REPO.parent / "stillhere" / "pilot" / "film" / "film_stim_v2" / "MANIFEST.sha256"


def consts(path):
    """The kernel's literal constants; STAGE is textwrap.dedent of a literal."""
    lit = lambda v: textwrap.dedent(ast.literal_eval(v.args[0])) if isinstance(v, ast.Call) else ast.literal_eval(v)
    return {n.targets[0].id: lit(n.value) for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in ("HF_PINS", "LOCK", "STITCH", "STAGE", "PIN")}


def unbound_globals(src, name):
    """Names a module reads (at module level or as globals from its functions) that it never binds and that are not builtins."""
    top = symtable.symtable(src, name, "exec"); bound = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported()}
    read, todo = set(), [top]
    while todo:
        t = todo.pop(); todo += t.get_children()
        for s in t.get_symbols():
            if s.is_referenced() and (t is top or s.is_global()) and not (t is not top and (s.is_assigned() or s.is_parameter())):
                read.add(s.get_name())
    return sorted(read - bound - set(dir(builtins)))


@pytest.mark.parametrize("version", list(KERNELS))
def test_every_name_a_film_kernel_reads_is_bound(version):
    assert unbound_globals(KERNELS[version].read_text(), version) == []


def test_the_unbound_name_check_catches_the_old_edit_kernels_bug():
    old = 'res = {}\njson_dump = print\njson_dump(res, f"tierD_{name}_forecast.json")\n'
    assert unbound_globals(old, "old_rem") == ["name"]


def test_edit_kernels_write_their_own_record_names():
    for v, rec in (("rem", "tierD_film_rem_forecast.json"), ("am", "tierD_film_am_forecast.json")):
        src = KERNELS[v].read_text(); assert src.count(rec) == 1 and "tierD_{name}" not in src
    assert "tierD_bridge_forecast.json" in KERNELS["intact"].read_text()


def test_one_lock_one_set_of_pins_and_the_reference_stitch_in_all_three():
    c = {v: consts(p) for v, p in KERNELS.items()}
    assert c["intact"]["LOCK"] == (FILM / "film_lock.txt").read_text() and all(c[v]["LOCK"] == c["intact"]["LOCK"] for v in c)
    assert all(c[v]["HF_PINS"] == c["intact"]["HF_PINS"] for v in c) and set(c["intact"]["HF_PINS"]) == {
        "facebook/tribev2", "facebook/vjepa2-vitg-fpc64-256", "facebook/w2v-bert-2.0", "meta-llama/Llama-3.2-3B"}
    lock = dict(l.split("==") for l in c["intact"]["LOCK"].splitlines() if "==" in l)
    assert {k: lock[k] for k in ("torch", "transformers", "accelerate", "moviepy")} == {"torch": "2.6.0", "transformers": "5.17.0",
                                                                                       "accelerate": "1.15.0", "moviepy": "2.2.1"}
    assert all(c[v]["STITCH"] == c["intact"]["STITCH"] for v in c) and c["intact"]["STITCH"].startswith("def stitched_predict(m, ev, aud_end):")
    if COMMON.exists():
        src = COMMON.read_text()
        ref = next(ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "stitched_predict")
        assert c["intact"]["STITCH"] == ref


def test_events_put_words_before_chunking_and_predict_through_the_harness():
    st = consts(KERNELS["intact"])["STAGE"]
    assert "get_events_dataframe(" not in st and "m.predict(" not in st and "from stitch import stitched_predict" in st
    i_words, i_chunk = st.index("ExtractWordsFromAudio(language=\"english\")"), st.index("ChunkEvents(event_type_to_chunk=\"Audio\"")
    assert st.index("ExtractAudioFromVideo()") < i_words < i_chunk < st.index("ChunkEvents(event_type_to_chunk=\"Video\"")
    assert 'out["words_ok"]' in st and "worst_shift_count" in st
    for v in ("rem", "am"):                                             # the edit runs load intact's events, so they share its words
        assert 'for stage in ("video", "audio", "text", "predict", "predict_audio"):' in KERNELS[v].read_text()


@pytest.mark.skipif(not (COMMON.exists() and STIM_MANIFEST.exists()), reason="needs the analysis tree beside the repository")
def test_the_builder_rebuilds_the_committed_kernels_byte_for_byte(tmp_path):
    r = subprocess.run([sys.executable, str(FILM / "make_film_kernels.py"), str(STIM_MANIFEST), str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    for v, p in KERNELS.items():
        assert hashlib.sha256((tmp_path / p.relative_to(FILM)).read_bytes()).hexdigest() == hashlib.sha256(p.read_bytes()).hexdigest(), v
        assert (tmp_path / v / "kernel-metadata.json").read_bytes() == (p.parent / "kernel-metadata.json").read_bytes(), v
    assert json.loads((tmp_path / "rem/kernel-metadata.json").read_text())["kernel_sources"] == ["redlotusthepotus/stillhere-pilot-bridge-intact"]
