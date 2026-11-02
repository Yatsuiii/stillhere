"""Files for reproducing Paranoia (R2 and R5) outside our runs (cloud review of Nov 2, 12:15 UTC, item 6): what scripts/reproduce_paranoia.sh
fetches and lays out, and what scripts/make_paranoia_release.sh stages. Every expected SHA-256 is read from the scorer and the runner as
committed (scripts/r2_paranoia.py: STIM_PINS, WORD_PINS, SURPRISAL_PIN, FORECAST_PINS; r2/run_r2_forecast.py: FULLTEXT_SHA;
results/r2_paranoia_SHA256.json for the atlas, which our scoring run read), never restated here. Standard library only; prints sizes,
counts and hashes, never word text.
  sizes DATA [--no-gpu]        what a run fetches, in bytes, and how long our runs took
  fetch-public DATA            the three ds001338 recordings, the full text and the Schaefer atlas, each checked
  fetch-release DEST [--forecasts]   our word tables and surprisal (and with --forecasts our nine stock forecasts) from the release
  common DATA                  DATA/k2/tribe_common.py from the R2 forecast kernel's embedded COMMON, after its SHA-256 check
  layout DATA                  the runner and the Alice harness it imports, copied into DATA's analysis-tree layout
  seed-words DATA SRC          our word tables as each full stage's stim.tsv, so TRIBE's speech-to-text does not run
  stage-release SRC OUT [--surprisal PATH] [--copy]   our forecasts and word tables checked against the pins, with SHA256SUMS and a README
usage: python3 scripts/paranoia_files.py COMMAND ..."""
import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORIES, MODES = ("story1", "story2", "story3"), ("audio", "full", "shuf")
STIM_URL = "https://s3.amazonaws.com/openneuro.org/ds001338/stimuli/"
ATLAS_URL = ("https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
             "Parcellations/FreeSurfer5.3/fsaverage5/label/")
ATLAS = [f"{h}.Schaefer2018_400Parcels_17Networks_order.annot" for h in ("lh", "rh")]
RELEASE_TAG = "paranoia-forecasts-v1"
RELEASE_URL = os.environ.get("STILLHERE_RELEASE_URL", f"https://github.com/Yatsuiii/stillhere/releases/download/{RELEASE_TAG}/")
COMMON_SHA = "8df0380c20b62a28263800e7ddcba0df805cbc4b919dbc4e702cccdfb1afd7ba"   # tribe_common as the R2 kernels ran it (rule section 10)
KERNEL = "r2/kaggle_forecast/a/stillhere_r2_forecast_a.py"


def assigned(rel, name):
    """A module-level constant of a committed file, read without importing it."""
    for n in ast.parse(open(os.path.join(REPO, rel)).read()).body:
        if not isinstance(n, ast.Assign):
            continue
        t = n.targets[0]
        if getattr(t, "id", None) == name:
            return ast.literal_eval(n.value)
        if isinstance(t, ast.Tuple) and isinstance(n.value, ast.Tuple):          # A, B = x, y
            for el, v in zip(t.elts, n.value.elts):
                if getattr(el, "id", None) == name:
                    return ast.literal_eval(v)
    raise SystemExit(f"{rel} assigns no {name}")


def pins() -> dict:
    """{file name: SHA-256} for every file a judge fetches or we publish, from the committed scorer, runner and our scoring record."""
    stim, words = assigned("scripts/r2_paranoia.py", "STIM_PINS"), assigned("scripts/r2_paranoia.py", "WORD_PINS")
    read = json.load(open(os.path.join(REPO, "results/r2_paranoia_SHA256.json")))
    out = {f"{s}_audio.wav": stim[s] for s in STORIES} | {f"words_{s}.tsv": words[s] for s in STORIES}
    out |= {f"pred_{k}.npz": v for k, v in assigned("scripts/r2_paranoia.py", "FORECAST_PINS").items()}
    out |= {"surprisal.json": assigned("scripts/r2_paranoia.py", "SURPRISAL_PIN"),
            "story_fulltext.txt": assigned("r2/run_r2_forecast.py", "FULLTEXT_SHA")} | {a: read[a] for a in ATLAS}
    return out


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, dst, want):
    """Download url to dst unless dst already holds the expected bytes; refuse (and remove the file) on any other SHA-256."""
    if not (os.path.isfile(dst) and sha(dst) == want):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"; urllib.request.urlretrieve(url, tmp); os.replace(tmp, dst)
    got = sha(dst)
    if got != want:
        os.remove(dst)
        raise SystemExit(f"refused: {os.path.basename(dst)} from {url} has SHA-256 {got[:16]}..., expected {want[:16]}...")
    print(f"{os.path.basename(dst)}: {os.path.getsize(dst):,} bytes, SHA-256 {want[:16]}... ok", flush=True)


def head(url) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get("Content-Length", 0))


def sizes(data, no_gpu):
    dc = json.load(open(os.path.join(REPO, "r2/data_check.json")))
    gii = sum(v[h]["bytes"] for v in dc["runs"].values() for h in ("L", "R"))
    stim = sum(head(STIM_URL + f) for f in [f"{s}_audio.wav" for s in STORIES] + ["story_fulltext.txt"])
    atlas = sum(head(ATLAS_URL + a) for a in ATLAS)
    ours = os.path.join(REPO, "results/r2_run/RUN.md")
    rel = 295_762_704 if no_gpu else 538_389                        # the staged release: all 13 files, or the word tables and surprisal
    rows = [("brain data: 132 fsaverage5 GIFTIs (fMRIPrep 22.0.2) and 3 events files", gii),
            ("stimuli: the 3 recordings and the full text", stim), ("atlas: Schaefer 400, 17 networks, fsaverage5", atlas),
            (("our nine stock forecasts, word tables and surprisal (release)" if no_gpu else "our word tables and surprisal (release)"), rel)]
    print(f"Paranoia (OpenNeuro ds001338, Finn et al. 2018, CC BY-NC 4.0) into {data}:")
    for name, b in rows:
        print(f"  {name:72} {b / 1e6:9,.1f} MB")
    print(f"  {'total':72} {sum(b for _, b in rows) / 1e6:9,.1f} MB")
    if not no_gpu:
        print("then 9 stock TRIBE v2 forecasts on your GPU (ours: 81 min in all on a Kaggle T4 with every model on the GPU; "
              "r2/run_r2_forecast.py keeps the sound and brain models on the CPU, so expect longer on a smaller GPU),")
    print("then scoring on the CPU (ours: 4,134 s for R2 and 99 s for R5 on a Kaggle CPU kernel; " + os.path.relpath(ours, REPO) + ").")


def fetch_public(data):
    p = pins()
    for f in [f"{s}_audio.wav" for s in STORIES] + ["story_fulltext.txt"]:
        fetch(STIM_URL + f, os.path.join(data, "r2/stimuli", f), p[f])
    for a in ATLAS:
        fetch(ATLAS_URL + a, os.path.join(data, "atlas", a), p[a])


def fetch_release(dest, forecasts):
    p = pins()
    names = [f"words_{s}.tsv" for s in STORIES] + ["surprisal.json"]
    names += [f"pred_stock_{s}_{k}.npz" for s in STORIES for k in MODES] if forecasts else []
    for f in names:
        fetch(RELEASE_URL + f, os.path.join(dest, f), p[f])


def common(data):
    """The R2 forecast kernel's embedded COMMON (tribe_common as the D22 kernels ran it), checked, as DATA/k2/tribe_common.py. Nothing is
    copied into the repository."""
    c = assigned(KERNEL, "COMMON")
    got = hashlib.sha256(c.encode()).hexdigest()
    if got != COMMON_SHA:
        raise SystemExit(f"refused: {KERNEL}'s COMMON has SHA-256 {got[:16]}..., expected {COMMON_SHA[:16]}...")
    os.makedirs(os.path.join(data, "k2"), exist_ok=True)
    open(os.path.join(data, "k2/tribe_common.py"), "w").write(c)
    print(f"k2/tribe_common.py: COMMON of {KERNEL}, SHA-256 {COMMON_SHA[:16]}... ok", flush=True)


def layout(data):
    """r2/run_r2_forecast.py runs from an analysis tree (r2/ beside alice/ and k2/); copy it and the harness it imports into DATA."""
    for rel in ("r2/run_r2_forecast.py", "alice/forecast/run_alice_forecast.py", "alice/forecast/run_alice_forecast_tuned.py"):
        os.makedirs(os.path.join(data, os.path.dirname(rel)), exist_ok=True)
        shutil.copyfile(os.path.join(REPO, rel), os.path.join(data, rel))
    print("runner and harness copied into", data, flush=True)


def seed_words(data, src):
    """Our word tables (checked) as stim.tsv in each full stage's work folder: run_r2_forecast.prepare reads a table already there instead
    of running TRIBE's speech-to-text, so the forecasts start from the same words as ours."""
    p = pins()
    for s in STORIES:
        f = os.path.join(src, f"words_{s}.tsv")
        if sha(f) != p[f"words_{s}.tsv"]:
            raise SystemExit(f"refused: {f} is not our word table")
        d = os.path.join(data, f"r2/forecast/tmp/{s}_full"); os.makedirs(d, exist_ok=True)
        shutil.copyfile(f, os.path.join(d, "stim.tsv"))
    print("our word tables placed as each full stage's stim.tsv (no speech-to-text)", flush=True)


README = """# Paranoia forecasts: stock TRIBE v2 on OpenNeuro ds001338 (StillHere R2 and R5)

The nine stock TRIBE v2 forecasts (audio-only, full, and full with the word texts shuffled, for each of the three story parts), the three
word tables they were made from (TRIBE v2's own speech-to-text) and the words' GPT-2 small surprisal, exactly as our scoring run read
them on Nov 1 2026. Each SHA-256 in SHA256SUMS equals the pin committed in scripts/r2_paranoia.py (FORECAST_PINS, WORD_PINS,
SURPRISAL_PIN) before any Paranoia brain data was downloaded. With these and the public scans, `scripts/reproduce_paranoia.sh --no-gpu
DATA_DIR` re-scores R2 and R5 without a GPU and prints the numbers beside ours.

Forecast, not a scan. Research use. Not a diagnosis or medical advice.

- TRIBE v2: Meta FAIR, d'Ascoli et al. 2026, CC BY-NC 4.0. These forecast arrays are TRIBE v2 outputs and are shared under the same
  licence (non-commercial use, with attribution).
- Stimuli and brain data: Finn, Corlett, Chen, Bandettini and Constable, Nature Communications 2018; OpenNeuro ds001338 v1.0.0
  (DOI 10.18112/openneuro.ds001338.v1.0.0), CC BY-NC 4.0. The word tables and surprisal are derived from its recordings.
- Each pred_*.npz holds P (seconds x 20,484 fsaverage5 vertices, float32) and starts (the second of each row).
"""


def stage_release(src, out, surprisal, copy):
    """Our forecasts and word tables (and surprisal), each checked against the committed pins, into OUT with SHA256SUMS and a README.
    Symbolic links by default (OUT may be in RAM); --copy copies."""
    p = pins()
    names = [f"pred_stock_{s}_{k}.npz" for s in STORIES for k in MODES] + [f"words_{s}.tsv" for s in STORIES]
    files = {f: os.path.join(src, f) for f in names}
    if surprisal:
        files["surprisal.json"] = surprisal
    bad = sorted(f for f, path in files.items() if sha(path) != p[f])
    if bad:
        raise SystemExit(f"refused: not our pinned files: {bad}")
    os.makedirs(out, exist_ok=True)
    for f, path in files.items():
        t = os.path.join(out, f)
        if os.path.lexists(t):
            os.remove(t)
        (shutil.copyfile if copy else os.symlink)(os.path.realpath(path), t)
    with open(os.path.join(out, "SHA256SUMS"), "w") as fh:
        for f in sorted(files):
            fh.write(f"{p[f]}  {f}\n")
    open(os.path.join(out, "README_RELEASE.md"), "w").write(README)
    print(f"{len(files)} files checked against the pins, {sum(os.path.getsize(v) for v in files.values()) / 1e6:,.1f} MB; staged in {out} "
          f"({'copies' if copy else 'symbolic links'}); release tag {RELEASE_TAG}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sizes"); s.add_argument("data"); s.add_argument("--no-gpu", action="store_true")
    sub.add_parser("fetch-public").add_argument("data")
    r = sub.add_parser("fetch-release"); r.add_argument("dest"); r.add_argument("--forecasts", action="store_true")
    sub.add_parser("common").add_argument("data")
    sub.add_parser("layout").add_argument("data")
    w = sub.add_parser("seed-words"); w.add_argument("data"); w.add_argument("src")
    g = sub.add_parser("stage-release"); g.add_argument("src"); g.add_argument("out"); g.add_argument("--surprisal")
    g.add_argument("--copy", action="store_true")
    a = ap.parse_args(argv)
    {"sizes": lambda: sizes(a.data, a.no_gpu), "fetch-public": lambda: fetch_public(a.data),
     "fetch-release": lambda: fetch_release(a.dest, a.forecasts), "common": lambda: common(a.data), "layout": lambda: layout(a.data),
     "seed-words": lambda: seed_words(a.data, a.src), "stage-release": lambda: stage_release(a.src, a.out, a.surprisal, a.copy)}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
