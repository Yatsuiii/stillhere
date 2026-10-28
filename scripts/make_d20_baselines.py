"""Builds the notthefall (D20) cheap-baseline kernel (docs/rules/cheap_baselines_rule.md; exploratory), pushed only after the frozen
stillhere-d20-score run is COMPLETE (docs/rules/cheap_baselines_thursday.md). OUT/kds is the code dataset stillhere-d20-baselines-code
(stillhere/cheap_baselines.py and stillhere/d20_baselines.py, flat, with their MANIFEST); OUT/kernel is the private CPU kernel
stillhere-d20-baselines, internet on for d20_keys.json's allow-list only, attaching that dataset, the frozen code dataset, the
speech-to-text prediction output and the frozen scoring output. The frozen stub and its code dataset are neither read nor written here.
usage: python scripts/make_d20_baselines.py OUT"""
import hashlib
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OWNER = "redlotusthepotus"
KERNEL, CODE = "stillhere-d20-baselines", "stillhere-d20-baselines-code"
STUB = '''"""stillhere-d20-baselines (PRIVATE Kaggle CPU kernel; exploratory; stillhere-app docs/rules/cheap_baselines_rule.md). Pushed only after
stillhere-d20-score is COMPLETE; refuses without its scored tierD_d20_score.json. Writes /kaggle/working/d20_baselines/tierD_d20_baselines.json."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(glob.glob("/kaggle/input/**/d20_baselines.py", recursive=True)[0]))
import d20_baselines
d20_baselines.main()
'''
META = {"id": f"{OWNER}/{KERNEL}", "title": KERNEL, "code_file": "stillhere_d20_baselines.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_internet": True,
        "dataset_sources": [f"{OWNER}/stillhere-d20-score-code", f"{OWNER}/{CODE}"], "competition_sources": [], "model_sources": [],
        "kernel_sources": [f"{OWNER}/stillhere-d20-pred-asr", f"{OWNER}/stillhere-d20-score"]}


def build(out):
    """Writes OUT/kds and OUT/kernel (replacing only those two folders); returns the code dataset's MANIFEST hash."""
    kds, ker = f"{out}/kds", f"{out}/kernel"
    for d in (kds, ker):
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for f in ("cheap_baselines.py", "d20_baselines.py"):
        shutil.copy(f"{REPO}/stillhere/{f}", f"{kds}/{f}")
    man = "".join(f"{hashlib.sha256(open(f'{kds}/{f}', 'rb').read()).hexdigest()}  {f}\n" for f in sorted(os.listdir(kds)))
    open(f"{kds}/MANIFEST.sha256", "w").write(man)
    json.dump({"title": CODE, "id": f"{OWNER}/{CODE}", "licenses": [{"name": "CC0-1.0"}]}, open(f"{kds}/dataset-metadata.json", "w"))
    open(f"{ker}/stillhere_d20_baselines.py", "w").write(STUB)
    json.dump(META, open(f"{ker}/kernel-metadata.json", "w"))
    return hashlib.sha256(man.encode()).hexdigest()


if __name__ == "__main__":
    print(build(sys.argv[1]))
