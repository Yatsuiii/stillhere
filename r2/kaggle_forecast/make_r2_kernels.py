"""Builds the two R2 stock forecast kernels from one template (r2_forecast_tpl.py, docs/rules/r2_paranoia_rule.md, amendment of Nov 1):
stillhere-r2-forecast-a (audio-only and full, story1-3; attaches stillhere-r2-asr's output and the HF token) and stillhere-r2-forecast-b
(shuffled, story1-3; attaches part A's output). Embeds byte for byte: the film kernels' package lock and Hugging Face pins
(kernels/film/), tribe_common as the D22 kernels ran it (COMMON of d22/d3/kernels/stock/d3_pred.py, SHA-256 asserted), the R2 runner's
check functions (r2/run_r2_forecast.py) and the stage script (r2_stage.py).
usage: python make_r2_kernels.py [OUT_DIR]"""
import ast
import hashlib
import json
import os
import sys

H = os.path.dirname(os.path.abspath(__file__)); PUB = os.path.dirname(os.path.dirname(H))
ANALYSIS = os.path.join(os.path.dirname(PUB), "stillhere")
D3 = os.path.join(ANALYSIS, "d22/d3/kernels/stock/d3_pred.py")                       # kept outside the repository
COMMON_SHA = "8df0380c20b62a28263800e7ddcba0df805cbc4b919dbc4e702cccdfb1afd7ba"
STIM_SHA = {"story1": "e0a674401cd64580aeb50327af4393332f387df87ddbf0a65d68a96eb55a6c3e",
            "story2": "e8b784671a748ad03f91227d1152d6219d51f713a46d6d6e41c63c17deb98e44",
            "story3": "a739113b62f1aef92a2ead0fa069d97c4e05e1478123d26f42d223b05f970034"}
WORDS_SHA = {"story1": "1a98288a1b8d48a468ce0bb6b18cc82a50db6a829d2c5fe0789b9339c185f8aa",          # stillhere-r2-asr (r2/kaggle_asr)
             "story2": "d3d07e7b93a35dfd44748c16e3f626021d98e6d573206658da5ca06f30b82bb9",
             "story3": "db9968150bc263c7d1f82f9cf88c31ec41e20f6e486b21e4d717768144a06e6f"}
OWNER = "redlotusthepotus"
OUT = sys.argv[1] if len(sys.argv) > 1 else H


def assigned(path, name):
    src = open(path).read()
    return next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == name)


def checks_module():
    """The runner's check functions, byte for byte, with the imports and constant they use."""
    src = open(os.path.join(PUB, "r2/run_r2_forecast.py")).read()
    fns = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    head = ('"""r2/run_r2_forecast.py\'s check functions, byte for byte."""\nimport csv\nimport re\nfrom collections import Counter\n\n'
            'STORIES = ("story1", "story2", "story3")\n')
    return head + "".join(f"\n\n{fns[k]}\n" for k in ("words_table", "shift_copies", "fulltext_counts", "check_shuffled"))


def build(part):
    common = assigned(D3, "COMMON"); assert hashlib.sha256(common.encode()).hexdigest() == COMMON_SHA, "d3 COMMON differs from its pin"
    film = os.path.join(PUB, "kernels/film")
    src = open(os.path.join(film, "make_film_kernels.py")).read()
    pins = next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "HF_PINS")
    s = open(os.path.join(H, "r2_forecast_tpl.py")).read()
    s = s.replace("@@PART@@", part)
    for k, v in (("STIM_SHA", STIM_SHA), ("WORDS_SHA", WORDS_SHA), ("HF_PINS", pins), ("LOCK", open(os.path.join(film, "film_lock.txt")).read()),
                 ("COMMON", common), ("COMMON_SHA", COMMON_SHA), ("CHECKS", checks_module()), ("STAGE", open(os.path.join(H, "r2_stage.py")).read())):
        assert s.count(f'"@@{k}@@"') == 1, k
        s = s.replace(f'"@@{k}@@"', repr(v))
    assert "@@" not in s
    ast.parse(s)
    slug = f"stillhere-r2-forecast-{part.lower()}"
    d = os.path.join(OUT, part.lower()); os.makedirs(d, exist_ok=True)
    code = f"stillhere_r2_forecast_{part.lower()}.py"
    open(os.path.join(d, code), "w").write(s)
    meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": code, "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "machine_shape": "NvidiaTeslaT4", "enable_internet": True, "dataset_sources": [f"{OWNER}/stillhere-hf"],
            "competition_sources": [], "model_sources": [],
            "kernel_sources": [f"{OWNER}/stillhere-r2-asr"] if part == "A" else [f"{OWNER}/stillhere-r2-forecast-a"]}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"))
    return os.path.join(d, code), hashlib.sha256(s.encode()).hexdigest()


if __name__ == "__main__":
    for p in ("A", "B"):
        f, h = build(p); print(os.path.relpath(f, OUT), h)
