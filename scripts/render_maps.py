"""Renders Schaefer-400 parcel values (or 20,484 fsaverage5 vertex values) as one image of four surface views: left and right hemisphere,
lateral and medial, on nilearn's bundled inflated fsaverage5 with sulcal depth underneath. Used for the film beat's maps (R = intact -
faces removed, C = intact - area matched, from results/face_map_RC.json; or a viewer map). --synthetic draws a smooth made-up map and
stamps "PLACEHOLDER: synthetic map" across it, so a placeholder can never pass for a result. Needs nilearn, nibabel and matplotlib;
nothing is downloaded (fsaverage5 ships with nilearn). Parcels outside the tested 374 can be greyed out with --tested-only.
usage: python scripts/render_maps.py OUT.png (--synthetic SEED | --rc FACE_MAP_RC.json --key R|C | --vertices NPY) [--atlas DIR]
       [--title TEXT] [--vmax X] [--tested-only] [--size W H]"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ATLAS = REPO.parent / "stillhere/atlas"                                  # lh/rh.Schaefer2018_400Parcels_17Networks_order.annot
NV = 10242                                                              # vertices per fsaverage5 hemisphere
PLACEHOLDER = "PLACEHOLDER: synthetic map"


def parcels(atlas=ATLAS) -> dict:
    from stillhere import face_map as FM
    return FM.all_parcels(*(f"{atlas}/{h}.Schaefer2018_400Parcels_17Networks_order.annot" for h in ("lh", "rh")))


def to_vertices(values, P: dict, tested_only: bool = False) -> np.ndarray:
    """400 parcel values -> 20,484 vertex values (NaN on the medial wall, and outside the tested parcels with tested_only)."""
    values = np.asarray(values, float)
    assert values.shape == (len(P["vertices"]),), f"expected {len(P['vertices'])} parcel values, got {values.shape}"
    out = np.full(2 * NV, np.nan)
    for j, v in enumerate(P["vertices"]):
        if not tested_only or P["tested"][j]:
            out[v] = values[j]
    return out


def synthetic(P: dict, seed: int) -> np.ndarray:
    """A smooth made-up parcel map: a few Gaussian bumps on the sphere, signed. For placeholder frames only."""
    from nilearn import datasets, surface
    fs5 = datasets.fetch_surf_fsaverage("fsaverage5")
    xyz = np.vstack([surface.load_surf_mesh(fs5[f"sphere_{h}"]).coordinates for h in ("left", "right")])
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    cents = np.array([xyz[v].mean(0) / np.linalg.norm(xyz[v].mean(0)) for v in P["vertices"]])
    rng = np.random.default_rng(seed); vals = np.zeros(len(cents))
    for _ in range(5):
        c = rng.normal(size=3); c /= np.linalg.norm(c)
        vals += rng.choice([-1, 1]) * rng.uniform(0.5, 1.0) * np.exp(-(1 - cents @ c) / 0.08)
    return vals / np.abs(vals).max()


def render(vert: np.ndarray, out: Path, title: str = "", vmax: float | None = None, placeholder: bool = False,
           size=(1600, 430)) -> Path:
    """Four views of 20,484 vertex values in one PNG of `size` pixels; symmetric diverging colour scale."""
    os.environ.setdefault("MPLCONFIGDIR", str(REPO.parent / ".cache/mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import datasets, plotting
    fs5 = datasets.fetch_surf_fsaverage("fsaverage5")
    vmax = float(vmax or np.nanmax(np.abs(vert)) or 1.0)
    dpi = 100; fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi, facecolor="white")
    views = [("left", "lateral"), ("left", "medial"), ("right", "medial"), ("right", "lateral")]
    for i, (h, view) in enumerate(views):
        ax = fig.add_axes([0.005 + i * 0.235, 0.02, 0.235, 0.84], projection="3d")
        data = vert[:NV] if h == "left" else vert[NV:]
        plotting.plot_surf_stat_map(fs5[f"infl_{h}"], data, hemi=h, view=view, bg_map=fs5[f"sulc_{h}"], axes=ax, figure=fig,
                                    cmap="RdBu_r", vmax=vmax, colorbar=False, threshold=None)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-vmax, vmax))
    cax = fig.add_axes([0.945, 0.15, 0.012, 0.6]); fig.colorbar(sm, cax=cax).ax.tick_params(labelsize=11)
    if title:
        fig.text(0.01, 0.93, title, fontsize=18, color="#1d1f23", weight="bold")
    if placeholder:
        fig.text(0.5, 0.45, PLACEHOLDER, fontsize=34, color="#4a3b00", ha="center", va="center", weight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#ffe58a", ec="#d4a500", alpha=0.92))
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white"); plt.close(fig)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("out", type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic", type=int, metavar="SEED"); g.add_argument("--rc", type=Path); g.add_argument("--vertices", type=Path)
    ap.add_argument("--key", choices=("R", "C")); ap.add_argument("--atlas", type=Path, default=ATLAS); ap.add_argument("--title", default="")
    ap.add_argument("--vmax", type=float); ap.add_argument("--tested-only", action="store_true")
    ap.add_argument("--size", type=int, nargs=2, default=(1600, 430)); a = ap.parse_args(argv)
    if a.vertices is not None:
        vert = np.load(a.vertices).astype(float)
        assert vert.shape == (2 * NV,), f"expected {2 * NV} vertex values, got {vert.shape}"
    else:
        P = parcels(a.atlas)
        if a.synthetic is not None:
            vals = synthetic(P, a.synthetic)
        else:
            assert a.key, "--rc needs --key R or C"
            vals = json.load(open(a.rc))[a.key]
        vert = to_vertices(vals, P, a.tested_only)
    render(vert, a.out, a.title, a.vmax, a.synthetic is not None, tuple(a.size))
    print(f"{a.out}: {'synthetic placeholder' if a.synthetic is not None else 'map'}, {int(np.isfinite(vert).sum())} vertices coloured")


if __name__ == "__main__":
    main()
