"""Builds the worker's own model folder: a Hugging Face hub cache holding ONLY the files the worker loads (TRIBE v2's checkpoint and
config, w2v-BERT 2.0, Llama 3.2 3B without its original/ copy), as hard links into the laptop's existing cache (no extra disk space;
the Windows partition's ntfs-3g supports hard links), with real files in snapshots/ instead of symlinks into blobs/. The sandboxed
service binds this folder read-only, so it never sees the rest of the cache (other models, tokens). Prints counts and sizes only.
Then audits it: weights, configs and tokenizers only (allowed file names), no dataset files.
usage: python3 worker/link_models.py SRC_HUB DEST_DIR          (DEST_DIR/hub is created)
       python3 worker/link_models.py --audit DEST_DIR [CODE_DIR ...]   (what the install script runs before binding them)"""
import json
import os
import sys
from pathlib import Path

MODELS = {                                   # repo: (revision, files)
    "facebook/tribev2": ("f894e783020944dcd96e5568550afe2aa9743f9f", ["best.ckpt", "config.yaml"]),
    "facebook/w2v-bert-2.0": ("da985ba0987f70aaeb84a80f2851cfac8c697a7b",
                              ["config.json", "model.safetensors", "preprocessor_config.json"]),
    "meta-llama/Llama-3.2-3B": ("13afe5124825b4f3751f836b40dafda64c1ed062",
                                ["config.json", "generation_config.json", "model-00001-of-00002.safetensors",
                                 "model-00002-of-00002.safetensors", "model.safetensors.index.json", "special_tokens_map.json",
                                 "tokenizer_config.json", "tokenizer.json"]),
}
ALLOWED_SUFFIXES = (".ckpt", ".yaml", ".json", ".safetensors")


def link(src_hub: Path, dest: Path) -> dict:
    hub = dest / "hub"; out = {}
    for repo, (rev, files) in MODELS.items():
        d = hub / ("models--" + repo.replace("/", "--")); snap = d / "snapshots" / rev
        snap.mkdir(parents=True, exist_ok=True); (d / "refs").mkdir(exist_ok=True); (d / "refs/main").write_text(rev)
        for f in files:
            src = (src_hub / ("models--" + repo.replace("/", "--")) / "snapshots" / rev / f).resolve(strict=True)
            t = snap / f
            if t.exists() and os.path.samefile(t, src):
                continue
            if t.exists():
                t.unlink()
            os.link(src, t)
        out[repo] = {"revision": rev, "files": len(files), "bytes": sum((snap / f).stat().st_size for f in files)}
    return out


def audit(dest: Path) -> dict:
    """Every file under DEST: a model file (allowed suffix) or a refs/main pointer; nothing else, no symlink."""
    bad, n, size = [], 0, 0
    for root, dirs, files in os.walk(dest):
        for f in files:
            p = Path(root) / f; n += 1; size += p.stat().st_size
            if p.is_symlink() or not (f.endswith(ALLOWED_SUFFIXES) or (p.parent.name == "refs" and f == "main")):
                bad.append(str(p.relative_to(dest)))
    return {"files": n, "bytes": size, "unexpected": bad}


MEDIA = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4", ".mov", ".mkv", ".avi", ".nii", ".nii.gz", ".gii", ".edf", ".bdf", ".fif")
PRIVATE_MARKS = ("ds003171", "stillhere-private", "sub-", "question_blocklist", "kaggle.json", "api_key")


def wheel_files(tree: Path) -> set:
    """Every path an installed distribution lists in its .dist-info/RECORD (files shipped by a published wheel)."""
    out = set()
    for rec in tree.glob("*.dist-info/RECORD"):
        for line in rec.read_text(errors="replace").splitlines():
            if line:
                out.add(os.path.normpath(line.split(",")[0]))
    return out


def audit_code(tree: Path) -> dict:
    """For the bound code folders (site-packages, the spaCy model): no audio, video or brain-imaging file except those a published wheel
    ships (its RECORD lists them, e.g. nibabel's and nilearn's test templates), nothing whose path carries a private-data mark, no
    symlink leaving the tree. Counts only."""
    media, shipped, marked, escaping, n, size = [], 0, [], [], 0, 0
    root = tree.resolve(); listed = wheel_files(tree)
    for r, dirs, files in os.walk(tree):
        for f in files + [d for d in dirs if (Path(r) / d).is_symlink()]:
            p = Path(r) / f; n += 1
            if p.is_symlink() and not p.resolve().is_relative_to(root):
                escaping.append(str(p.relative_to(tree)))
            if f.lower().endswith(MEDIA):
                rel = os.path.normpath(str(p.relative_to(tree)))
                if rel in listed:
                    shipped += 1
                else:
                    media.append(rel)
            if any(m in str(p.relative_to(tree)) for m in PRIVATE_MARKS):
                marked.append(str(p.relative_to(tree)))
            if p.is_file():
                size += p.stat().st_size
    return {"files": n, "bytes": size, "media_files_from_wheels": shipped, "media_files": media, "private_marks": marked,
            "symlinks_leaving_tree": escaping}


if __name__ == "__main__":
    if sys.argv[1] == "--audit":                                  # --audit MODELS_DIR [CODE_DIR ...]: exit 1 on anything unexpected
        rep = {"models": audit(Path(sys.argv[2])), **{d: audit_code(Path(d)) for d in sys.argv[3:]}}
        print(json.dumps({k: {kk: (vv if not isinstance(vv, list) else vv[:20]) for kk, vv in v.items()} for k, v in rep.items()}, indent=1))
        bad = rep["models"]["unexpected"] or any(v["media_files"] or v["private_marks"] or v["symlinks_leaving_tree"]
                                                 for k, v in rep.items() if k != "models")
        sys.exit(1 if bad else 0)
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    print(json.dumps({"linked": link(src, dest), "audit": audit(dest)}, indent=1))
