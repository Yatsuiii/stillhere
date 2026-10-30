"""Builds the README's two images, docs/assets/stillhere-hero.png (also the hosted page's header, web/public/hero.png) and
docs/assets/stillhere-architecture.png, in the page's dark line-art style. Display only: no statistic, no new number.

- Hero, panel A: the menu's two brain areas on a contour-line left cortex (nilearn's bundled fsaverage5 pial surface, sulcal-depth
  isolines). Areas exactly as stillhere/heldout.py parcel_sets defines them (Schaefer-400, 17 networks): HEARING = SomMotB_Aud,
  LANGUAGE = left TempPar, DefaultB_Temp, DefaultB_PFCv, DefaultB_PFCl. Areas of interest only; no data values.
- Hero, panel B: the LPP ghost line from results/ghostline_lpp_s8.json under docs/rules/display_rule_ghostline.md: the gray dashed
  forecast in full first, then the short chip, then the coloured 49-listener line; the file's own labels and title; x in seconds of
  the section's scan; no other number.
- Architecture: the stations, the evidence path and the film path, in the submission's wording (docs/submission.md sections 5 to 7).

Needs nilearn, nibabel, matplotlib and scipy (panel A; nothing is downloaded) and rsvg-convert with the Noto Serif and Noto Sans Mono
fonts (the PNGs). Prints only file names and sizes.
usage: python scripts/build_readme_art.py [--atlas DIR] [--out DIR]"""
import argparse
import base64
import io
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INK, DIM, MUTE, BG, AMB, LINE = "#ece9e2", "#8c8881", "#bdb9b1", "#1d1d1d", "#e8a33d", "#3a3835"   # amber: real people only
SERIF, MONO = 'font-family="Noto Serif, Georgia, serif"', 'font-family="Noto Sans Mono, ui-monospace, monospace"'
S = f'fill="none" stroke="{INK}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"'
EPIGRAPH = "“Begin at the beginning, and go on till you come to the end: then stop.”"          # Alice, on the story menu


def T(x, y, size, text, extra=""):
    return f'<text x="{x}" y="{y}" font-size="{size}" {extra}>{text}</text>'


def crop_marks(w, h):
    return "".join(f'<path fill="none" stroke="{DIM}" stroke-width="1" d="M{x},{y + 16 * dy} V{y} H{x + 16 * dx}"/>'
                   for x, y, dx, dy in ((24, 24, 1, 1), (w - 24, 24, -1, 1), (24, h - 24, 1, -1), (w - 24, h - 24, -1, -1)))


# ---- glyphs: one per station and per part of the pipeline -------------------------------------------------------------------------
def glyph(kind, cx, cy):
    g = []
    if kind == "plan":                                       # a page with lines and a tick box
        g += [f'<path {S} d="M{cx-22},{cy-30} h32 l12,12 v48 h-44 z M{cx+10},{cy-30} v12 h12"/>',
              f'<path {S} d="M{cx-14},{cy-8} h26 M{cx-14},{cy+2} h26 M{cx-14},{cy+12} h16"/>',
              f'<rect {S} x="{cx-14}" y="{cy+19}" width="8" height="8"/>', f'<path {S} d="M{cx-13},{cy+23} l3,3 l6,-7"/>']
    elif kind == "build":                                    # a waveform cut at sentence boundaries
        for i, hh in enumerate((8, 16, 26, 12, 20, 30, 10, 18, 24, 14, 22, 8)):
            x = cx - 30 + i * 5.4 + (6 if i >= 4 else 0) + (6 if i >= 8 else 0)
            g.append(f'<path {S} d="M{x:.1f},{cy-hh/2} v{hh}"/>')
        g += [f'<path fill="none" stroke="{DIM}" stroke-width="1.3" stroke-dasharray="2 3" d="M{x:.1f},{cy-30} v60"/>' for x in (cx - 6.5, cx + 21)]
    elif kind == "audit":                                    # a lens over a plan
        g += [f'<path {S} d="M{cx-26},{cy-22} h30 M{cx-26},{cy-12} h22 M{cx-26},{cy-2} h14"/>', f'<circle {S} cx="{cx+4}" cy="{cy+4}" r="15"/>',
              f'<path fill="none" stroke="{INK}" stroke-width="3.2" stroke-linecap="round" d="M{cx+15},{cy+15} l14,14"/>',
              f'<path fill="none" stroke="{DIM}" stroke-width="1.2" d="M{cx-3},{cy-1} a8,8 0 0 1 9,-6"/>']
    elif kind == "forecast":                                 # a head in profile, sound in
        g += [f'<path {S} d="M{cx-4},{cy+30} v-10 c-18,-2 -24,-14 -22,-28 c2,-16 16,-26 32,-24 c16,2 26,14 24,30 c-1,6 -2,9 2,14 l-6,3 '
              f'c0,6 -2,10 -8,10 h-6 v5"/>',
              f'<path fill="none" stroke="{DIM}" stroke-width="1.3" d="M{cx-8},{cy-12} c5,-6 14,-6 18,0 M{cx-10},{cy-2} c6,-5 14,-3 18,2 '
              f'M{cx-4},{cy+8} c4,-3 9,-2 11,1"/>',
              f'<path fill="none" stroke="{DIM}" stroke-width="1.3" d="M{cx-40},{cy-6} q4,-6 8,0 t8,0 M{cx-40},{cy+4} q4,-6 8,0 t8,0"/>']
    elif kind == "evidence":                                 # forecast (dashed) against real listeners (solid, colour)
        g += [f'<path {S} d="M{cx-28},{cy-26} v52 h56"/>',
              f'<path fill="none" stroke="{DIM}" stroke-width="1.6" stroke-dasharray="4 3" d="M{cx-24},{cy+10} c8,-28 14,-28 20,-6 s12,18 20,-10 s8,-12 10,-4"/>',
              f'<path fill="none" stroke="{AMB}" stroke-width="1.7" d="M{cx-24},{cy+14} c8,-30 16,-26 20,-8 s12,22 20,-12 s8,-8 10,-2"/>']
    elif kind in ("film", "reel"):                           # a film frame; "film" shows a filled face and a same-size control
        g.append(f'<rect {S} x="{cx-30}" y="{cy-24}" width="60" height="48"/>')
        for i in range(6):
            x = cx - 26 + i * 10
            g += [f'<rect fill="none" stroke="{INK}" stroke-width="1.2" x="{x}" y="{y}" width="4" height="4"/>' for y in (cy - 21, cy + 17)]
        if kind == "film":
            g += [f'<ellipse cx="{cx-9}" cy="{cy}" rx="8" ry="10" fill="{DIM}" stroke="{INK}" stroke-width="1.4"/>',
                  f'<rect x="{cx+6}" y="{cy-8}" width="14" height="16" fill="none" stroke="{INK}" stroke-width="1.3" stroke-dasharray="2 2"/>']
        else:
            g += [f'<path {S} d="M{cx-22},{cy+12} l12,-14 l8,8 l6,-6 l14,12"/>',
                  f'<circle cx="{cx+12}" cy="{cy-6}" r="3" fill="none" stroke="{INK}" stroke-width="1.3"/>']
    elif kind == "request":
        g += [f'<rect {S} x="{cx-30}" y="{cy-18}" width="60" height="36" rx="2"/>', f'<path {S} d="M{cx-20},{cy-6} h30 M{cx-20},{cy+4} h18"/>',
              f'<path fill="none" stroke="{INK}" stroke-width="1.8" d="M{cx+3},{cy-2} v12"/>']
    elif kind == "egress":                                   # an allowlist sieve between two posts
        g.append(f'<path {S} d="M{cx-26},{cy+28} V{cy-26} M{cx+26},{cy+28} V{cy-26} M{cx-32},{cy-26} H{cx+32}"/>')
        g += [f'<circle cx="{cx-15+c*10}" cy="{cy-12+r*10}" r="{2.2 if (r + c) % 3 else 3.4}" fill="none" stroke="{INK}" stroke-width="1.2"/>'
              for r in range(4) for c in range(4)]
    elif kind == "laptop":
        g += [f'<rect {S} x="{cx-26}" y="{cy-22}" width="52" height="34" rx="2"/>', f'<path {S} d="M{cx-34},{cy+18} H{cx+34} l-6,6 H{cx-28} z"/>',
              f'<path fill="none" stroke="{DIM}" stroke-width="1.4" stroke-dasharray="4 3" d="M{cx-18},{cy-2} c6,-12 10,-12 14,0 s8,10 14,-4"/>']
    elif kind == "screen":                                   # the end screen: the forecast alone
        g += [f'<rect {S} x="{cx-30}" y="{cy-22}" width="60" height="40"/>', f'<path {S} d="M{cx-10},{cy+26} h20 M{cx},{cy+18} v8"/>',
              f'<path fill="none" stroke="{DIM}" stroke-width="1.6" stroke-dasharray="4 3" d="M{cx-22},{cy+4} c6,-16 12,-16 18,-2 s10,10 18,-8"/>']
    elif kind == "receipt":
        g += [f'<path {S} d="M{cx-20},{cy-28} h40 v52 l-5,-4 l-5,4 l-5,-4 l-5,4 l-5,-4 l-5,4 l-5,-4 l-5,4 z"/>',
              f'<path {S} d="M{cx-12},{cy-16} h24 M{cx-12},{cy-6} h24 M{cx-12},{cy+4} h14"/>']
    elif kind == "rules":                                    # a page with a lock: committed before the data
        g += [f'<path {S} d="M{cx-22},{cy-30} h32 l12,12 v48 h-44 z M{cx+10},{cy-30} v12 h12"/>', f'<path {S} d="M{cx-14},{cy-10} h26 M{cx-14},{cy} h18"/>',
              f'<rect {S} x="{cx-8}" y="{cy+10}" width="16" height="12"/>', f'<path {S} d="M{cx-5},{cy+10} v-4 a5,5 0 0 1 10,0 v4"/>']
    elif kind == "gpu":
        g += [f'<rect {S} x="{cx-22}" y="{cy-22}" width="44" height="44"/>', f'<rect {S} x="{cx-10}" y="{cy-10}" width="20" height="20"/>']
        g += [f'<path fill="none" stroke="{INK}" stroke-width="1.3" d="M{cx+o},{cy-22} v-7 M{cx+o},{cy+22} v7 M{cx-22},{cy+o} h-7 M{cx+22},{cy+o} h7"/>'
              for o in (-15, -5, 5, 15)]
    elif kind == "scans":                                    # a stack of scan volumes and one real trace
        g += [f'<path {S} d="M{cx-26},{cy-20+i*14} l14,-8 h38 l-14,8 z"/>' for i in range(3)]
        g.append(f'<path fill="none" stroke="{AMB}" stroke-width="1.6" d="M{cx-14},{cy+12} c4,-6 8,-6 12,0 s8,6 12,0"/>')
    elif kind == "mask":
        g += [f'<rect {S} x="{cx-30}" y="{cy-24}" width="60" height="48"/>', f'<path {S} d="M{cx-30},{cy+24} l12,-12 M{cx+30},{cy-24} l-12,12"/>',
              f'<ellipse cx="{cx}" cy="{cy}" rx="11" ry="14" fill="none" stroke="{INK}" stroke-width="1.6" stroke-dasharray="3 2"/>']
    elif kind == "results":
        g.append(T(cx, cy + 12, 36, "{ }", f'{SERIF} fill="{INK}" text-anchor="middle"'))
    return "".join(g)


# ---- panel A: the menu's brain areas on a contour-line cortex -------------------------------------------------------------------
def brain_png(atlas: Path) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import nibabel.freesurfer as fs
    import numpy as np
    from matplotlib.colors import ListedColormap
    from nilearn import datasets, surface
    from scipy.ndimage import binary_closing, binary_fill_holes
    labels, _, names = fs.read_annot(str(atlas / "lh.Schaefer2018_400Parcels_17Networks_order.annot"))
    names = [n.decode() for n in names]
    area = np.zeros(labels.shape, int)                                     # 1 = HEARING, 2 = LANGUAGE (heldout.parcel_sets)
    for i, n in enumerate(names):
        if "SomMotB_Aud" in n:
            area[labels == i] = 1
        if any(p in n for p in ("TempPar", "DefaultB_Temp", "DefaultB_PFCv", "DefaultB_PFCl")):
            area[labels == i] = 2
    fs5 = datasets.fetch_surf_fsaverage("fsaverage5")
    mesh = surface.load_surf_mesh(fs5["pial_left"])
    v, f = np.asarray(mesh.coordinates), np.asarray(mesh.faces)
    sulc = surface.load_surf_data(fs5["sulc_left"])
    normal = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vis = f[normal[:, 0] < 0]                                              # faces turned toward a viewer on the left
    sx, sy = -v[:, 1], v[:, 2]                                             # lateral view, anterior to the left
    tri = mtri.Triangulation(sx, sy, vis)
    fig = plt.figure(figsize=(7.2, 5.0), dpi=200); fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_aspect("equal"); ax.patch.set_alpha(0)
    for k, tone in ((2, "#57534d"), (1, "#a9a49a")):                       # solid tones: an alpha fill shows the mesh seams
        sel = vis[(area[vis] == k).all(axis=1)]
        ax.tripcolor(mtri.Triangulation(sx, sy, sel), np.ones(len(sel)), cmap=ListedColormap([tone]), shading="flat",
                     edgecolors="face", linewidth=.3)
    ax.tricontour(tri, sulc, levels=np.linspace(np.percentile(sulc, 10), np.percentile(sulc, 90), 9), colors=INK, linewidths=.5,
                  linestyles="solid", alpha=.55)
    ax.autoscale_view(); ax.margins(.045); xl, yl = ax.get_xlim(), ax.get_ylim()
    g = plt.figure(figsize=(7.2, 5.0), dpi=100); gx = g.add_axes([0, 0, 1, 1]); gx.set_axis_off()   # silhouette: rasterize, one contour
    gx.tripcolor(tri, np.ones(len(vis)), cmap="gray_r", vmin=0, vmax=1, shading="flat", edgecolors="face"); gx.set_xlim(xl); gx.set_ylim(yl)
    g.canvas.draw(); img = binary_fill_holes(binary_closing(np.asarray(g.canvas.buffer_rgba())[::-1, :, 0] < 128, iterations=3)); plt.close(g)
    ax.contour(np.linspace(*xl, img.shape[1]), np.linspace(*yl, img.shape[0]), img.astype(float), levels=[.5], colors=INK, linewidths=1.3)
    ax.set_xlim(xl); ax.set_ylim(yl)
    buf = io.BytesIO(); fig.savefig(buf, format="png", transparent=True, dpi=125); plt.close(fig)
    return buf.getvalue()


# ---- the hero ------------------------------------------------------------------------------------------------------------------
def hero_svg(ghost: dict, brain: bytes) -> str:
    W, H = 1400, 560
    t, L = ghost["time_s"], ghost["lines"]
    X0, X1, Y0, Y1 = 792, 1336, 412, 474
    vals = [y for k in ("forecast", "group") for y in L[k]["values"] if y is not None]
    lo, hi = min(vals), max(vals)
    X = lambda x: X0 + (X1 - X0) * (x - t[0]) / (t[-1] - t[0])
    Y = lambda y: Y1 - (Y1 - Y0) * (y - lo) / (hi - lo)

    def line(k):                                                           # gaps (null) stay gaps; no smoothing
        segs, cur = [], []
        for x, y in zip(t, L[k]["values"]):
            if y is None:
                segs.append(cur); cur = []
            else:
                cur.append(f"{X(x):.1f},{Y(y):.1f}")
        return [c for c in segs + [cur] if len(c) > 1]
    P = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="StillHere. A: the brain areas on the menu. '
         f'B: {ghost["title"]}.">', f'<rect width="{W}" height="{H}" fill="{BG}"/>', crop_marks(W, H),
         T(700, 46, 15, EPIGRAPH, f'{SERIF} font-style="italic" fill="{MUTE}" text-anchor="middle"'),
         T(70, 176, 82, "STILLHERE", f'{SERIF} font-style="italic" font-weight="700" letter-spacing="7" fill="none" stroke="{INK}" stroke-width="1.6"'),
         T(74, 228, 19, "PILOT THE STORY BEFORE BOOKING THE SCANNER", f'{SERIF} letter-spacing="3.2" fill="{INK}"'),
         T(74, 268, 17, "Plan a listening study, build and check the stimulus,", f'{SERIF} font-style="italic" fill="{MUTE}"'),
         T(74, 296, 17, "forecast the brain response, then see how forecasts did on real scans", f'{SERIF} font-style="italic" fill="{MUTE}"')]
    key = (("STORY", f'<path fill="none" stroke="{INK}" stroke-width="1.5" d="M-16,0 q4,-8 8,0 t8,0 t8,0 t8,0"/>', "LibriVox · public"),
           ("FORECAST", f'<path fill="none" stroke="{DIM}" stroke-width="2" stroke-dasharray="5 4" d="M-18,0 h36"/>', "TRIBE v2, gray"),
           ("LISTENERS", f'<path fill="none" stroke="{AMB}" stroke-width="2.2" d="M-18,0 h36"/>', "public scans"),
           ("AREA", f'<rect x="-9" y="-6" width="18" height="12" fill="#57534d" stroke="{INK}" stroke-width=".8"/>', "on the menu"))
    for i, (lab, mark, cap) in enumerate(key):
        x = 120 + i * 160
        P += [T(x, 352, 11, lab, f'{MONO} letter-spacing="1.8" fill="{MUTE}" text-anchor="middle"'), f'<g transform="translate({x},372)">{mark}</g>',
              T(x, 400, 15, cap, f'{SERIF} font-style="italic" fill="{MUTE}" text-anchor="middle"')]
    P.append(T(74, 482, 13.5, "· Lewis Carroll, Alice’s Adventures in Wonderland (1865)", f'{SERIF} font-style="italic" fill="{DIM}"'))
    bx, by, bw, bh = 772, 92, 400, 278                                     # A
    P += [T(760, 82, 13, "A", f'{MONO} font-weight="700" fill="{INK}"'), T(780, 82, 11, "BRAIN AREAS ON THE MENU", f'{MONO} letter-spacing="1.8" fill="{MUTE}"'),
          f'<image href="data:image/png;base64,{base64.b64encode(brain).decode()}" x="{bx}" y="{by}" width="{bw}" height="{bh}"/>']
    for (px, py), (lx, ly), name, sub in (((bx + .60 * bw, by + .70 * bh), (1196, 288), "LANGUAGE AREAS", "left TempPar, DefaultB"),
                                          ((bx + .55 * bw, by + .555 * bh), (1196, 214), "HEARING AREAS", "SomMotB_Aud")):
        P += [f'<circle cx="{px:.0f}" cy="{py:.0f}" r="3" fill="{INK}"/>',
              f'<path fill="none" stroke="{DIM}" stroke-width="1" stroke-dasharray="2 3" d="M{px:.0f},{py:.0f} L{lx - 40},{ly - 4} H{lx - 6}"/>',
              T(lx, ly, 11, name, f'{MONO} letter-spacing="1.4" fill="{INK}"'), T(lx, ly + 18, 12.5, sub, f'{SERIF} font-style="italic" fill="{DIM}"')]
    P += [T(1196, 342, 12.5, "fsaverage5 surface", f'{SERIF} font-style="italic" fill="{DIM}"'),
          T(1196, 360, 12.5, "Schaefer-400, 17 networks", f'{SERIF} font-style="italic" fill="{DIM}"'),
          T(760, 400, 13, "B", f'{MONO} font-weight="700" fill="{INK}"'),                                  # B, in the rule's draw order
          T(1336, 400, 12.5, ghost["title"], f'{SERIF} font-style="italic" fill="{DIM}" text-anchor="end"')]
    P += [f'<polyline fill="none" stroke="{DIM}" stroke-width="1.5" stroke-dasharray="5 4" points="{" ".join(s)}"/>' for s in line("forecast")]
    P.append(T(780, 400, 11, ghost["chip"]["short"].upper(), f'{MONO} letter-spacing="1.8" fill="{MUTE}"'))
    P += [f'<polyline fill="none" stroke="{AMB}" stroke-width="1.9" stroke-linejoin="round" points="{" ".join(s)}"/>' for s in line("group")]
    P.append(f'<path stroke="{DIM}" stroke-width="1" d="M{X0},486 H{X1}"/>')
    for s in range(60, int(t[-1]) + 1, 60):
        P += [f'<path stroke="{DIM}" stroke-width="1" d="M{X(s):.1f},486 v5"/>', T(round(X(s), 1), 504, 10.5, str(s), f'{MONO} fill="{DIM}" text-anchor="middle"')]
    lab_f, lab_g = L["forecast"]["label"], L["group"]["label"]
    P += [T(X1, 530, 10.5, "seconds of scan", f'{MONO} fill="{DIM}" text-anchor="end"'),
          f'<path fill="none" stroke="{DIM}" stroke-width="1.5" stroke-dasharray="5 4" d="M{X0},526 h28"/>',
          T(X0 + 36, 530, 12.5, lab_f, f'{SERIF} font-style="italic" fill="{MUTE}"'),
          f'<path fill="none" stroke="{AMB}" stroke-width="1.9" d="M{X0 + 200},526 h28"/>',
          T(X0 + 236, 530, 12.5, lab_g, f'{SERIF} font-style="italic" fill="{MUTE}"'), "</svg>"]
    return "".join(P)


# ---- the architecture ----------------------------------------------------------------------------------------------------------
XS = (130, 400, 670, 940, 1210)                                            # five columns


def chip(x, y, text):
    """A model label in a box, as on an instrument panel."""
    w = 7.6 * len(text) + 16
    return (f'<rect x="{x - w / 2:.0f}" y="{y - 13}" width="{w:.0f}" height="19" fill="none" stroke="{DIM}" stroke-width="1"/>'
            + T(x, y + 1, 10.5, text, f'{MONO} letter-spacing="1.2" fill="{INK}" text-anchor="middle"'))


def node(x, y, kind, title, lines, num=None, model=None):
    """Glyph, title, an optional model chip, then the italic lines; returns (svg, y of the last line)."""
    g = [glyph(kind, x, y), T(x, y + 64, 16.5, (f"{num} · " if num else "") + title, f'{SERIF} font-weight="700" fill="{INK}" text-anchor="middle"')]
    y0 = y + 86
    if model:
        g.append(chip(x, y0 + 2, model)); y0 += 28
    g += [T(x, y0 + i * 19, 13.5, s, f'{SERIF} font-style="italic" fill="{MUTE}" text-anchor="middle"') for i, s in enumerate(lines)]
    return "".join(g), y0 + (len(lines) - 1) * 19


def check(x, y, question, outcome, branch="NO"):
    """A diamond under a node: the question above it, and where the branch goes (on NO the path stops there and says why)."""
    g = [T(x, y - 24, 12.5, question, f'{SERIF} font-style="italic" fill="{INK}" text-anchor="middle"'),
         f'<path fill="{BG}" stroke="{INK}" stroke-width="1.4" d="M{x},{y - 12} L{x + 12},{y} L{x},{y + 12} L{x - 12},{y} z"/>',
         f'<path fill="none" stroke="{DIM}" stroke-width="1" marker-end="url(#ahd)" d="M{x + 14},{y} H{x + 44}"/>',
         T(x + 50, y - 8, 10.5, branch, f'{MONO} letter-spacing="1.2" fill="{DIM}"')]
    g += [T(x + 50, y + 8 + i * 16, 12.5, s, f'{SERIF} font-style="italic" fill="{MUTE}"') for i, s in enumerate(outcome)]
    return "".join(g)


def arrow(d, dash=False):
    dashes = ' stroke-dasharray="5 4"' if dash else ""
    return f'<path fill="none" stroke="{INK}" stroke-width="1.3"{dashes} marker-end="url(#ah)" d="{d}"/>'


def section(y, label):
    return T(46, y, 12.5, label, f'{MONO} letter-spacing="3" fill="{INK}"') + f'<path stroke="{LINE}" stroke-width="1" d="M46,{y + 12} H1354"/>'


def chain(y, dashed_into=()):
    return [arrow(f"M{a + 50},{y} L{b - 50},{y}", b in dashed_into) for a, b in zip(XS, XS[1:])]


def ghost_band(ghost, y):
    """The ghost line between the stations and the evidence, in display_rule_ghostline's order: the gray dashed forecast in full,
    then the short chip, then the coloured 49-listener line; the file's labels and title; x in seconds of the section's scan."""
    t, L = ghost["time_s"], ghost["lines"]
    X0, X1, top, bot = 330, 1340, y + 34, y + 104
    vals = [v for k in ("forecast", "group") for v in L[k]["values"] if v is not None]
    lo, hi = min(vals), max(vals)
    X = lambda s: X0 + (X1 - X0) * (s - t[0]) / (t[-1] - t[0])
    Y = lambda v: bot - (bot - top) * (v - lo) / (hi - lo)
    seg = lambda k: [" ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in run) for run in _runs(t, L[k]["values"])]
    g = [T(46, y + 14, 12.5, "THE GHOST LINE", f'{MONO} letter-spacing="3" fill="{INK}"'),
         T(46, y + 44, 14, "“Gray is the forecast;", f'{SERIF} font-style="italic" fill="{MUTE}"'),
         T(46, y + 64, 14, "colour is real listeners.”", f'{SERIF} font-style="italic" fill="{MUTE}"'),
         T(46, y + 96, 12.5, "how station 5 shows a forecast", f'{SERIF} font-style="italic" fill="{DIM}"'),
         T(46, y + 114, 12.5, "against real listeners", f'{SERIF} font-style="italic" fill="{DIM}"'),
         T(X1, y + 14, 12.5, ghost["title"], f'{SERIF} font-style="italic" fill="{DIM}" text-anchor="end"')]
    g += [f'<polyline fill="none" stroke="{DIM}" stroke-width="1.5" stroke-dasharray="5 4" points="{p}"/>' for p in seg("forecast")]
    g.append(T(X0, y + 14, 11, ghost["chip"]["short"].upper(), f'{MONO} letter-spacing="1.8" fill="{MUTE}"'))
    g += [f'<polyline fill="none" stroke="{AMB}" stroke-width="1.8" stroke-linejoin="round" points="{p}"/>' for p in seg("group")]
    g += [f'<path stroke="{DIM}" stroke-width="1" d="M{X0},{bot + 10} H{X1}"/>',
          f'<path fill="none" stroke="{DIM}" stroke-width="1.5" stroke-dasharray="5 4" d="M{X0},{bot + 50} h28"/>',
          T(X0 + 36, bot + 54, 12.5, L["forecast"]["label"], f'{SERIF} font-style="italic" fill="{MUTE}"'),
          f'<path fill="none" stroke="{AMB}" stroke-width="1.8" d="M{X0 + 200},{bot + 50} h28"/>',
          T(X0 + 236, bot + 54, 12.5, L["group"]["label"], f'{SERIF} font-style="italic" fill="{MUTE}"'),
          T(X1, bot + 54, 10.5, "seconds of scan", f'{MONO} fill="{DIM}" text-anchor="end"')]
    for s in range(60, int(t[-1]) + 1, 60):
        g += [f'<path stroke="{DIM}" stroke-width="1" d="M{X(s):.1f},{bot + 10} v5"/>', T(round(X(s), 1), bot + 26, 10, str(s), f'{MONO} fill="{DIM}" text-anchor="middle"')]
    return "".join(g)


def _runs(t, values):                                                      # gaps (null) stay gaps
    out, cur = [], []
    for s, v in zip(t, values):
        if v is None:
            out.append(cur); cur = []
        else:
            cur.append((s, v))
    return [c for c in out + [cur] if len(c) > 1]


def architecture_svg(ghost: dict) -> str:
    W, H = 1400, 2140
    P = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="StillHere architecture">',
         f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
         f'<path d="M0,0 L10,5 L0,10" fill="none" stroke="{INK}" stroke-width="1.5"/></marker>'
         f'<marker id="ahd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
         f'<path d="M0,0 L10,5 L0,10" fill="none" stroke="{DIM}" stroke-width="1.5"/></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="{BG}"/>', crop_marks(W, H),
         T(700, 54, 15, "“Gray is the forecast; colour is real listeners.”", f'{SERIF} font-style="italic" fill="{MUTE}" text-anchor="middle"'),
         T(700, 118, 52, "STILLHERE", f'{SERIF} font-style="italic" font-weight="700" letter-spacing="5" fill="none" stroke="{INK}" stroke-width="1.3" text-anchor="middle"'),
         T(700, 156, 15, "ARCHITECTURE · FROM A TYPED REQUEST TO A CHECKED FORECAST", f'{SERIF} letter-spacing="3" fill="{INK}" text-anchor="middle"'),
         T(700, 184, 14.5, "Every model call passes one typed egress layer; every pre-registered test is scored by a rule committed before its brain data was read",
           f'{SERIF} font-style="italic" fill="{MUTE}" text-anchor="middle"'),
         section(236, "I · THE STATIONS · HOSTED PAGE, KEYS ON THE SERVER ONLY")]
    # I, row 1: request, egress, plan, build, audit
    y = 320
    parts = [node(XS[0], y, "request", "Request", ["a researcher types", "the study they want,", "never stored or logged"]),
             node(XS[1], y, "egress", "Typed egress", ["the only way to a model API:", "public text, scanned requests,", "allowlisted audio and frames"]),
             node(XS[2], y, "plan", "Plan", ["a strict study spec", "from the tested menu"], 1, "MUSE SPARK 1.3 · OPEN"),
             node(XS[3], y, "build", "Build and check", ["cut at sentence boundaries,", "each condition read back"], 2, "MUSE VOICE TRANSCRIBE"),
             node(XS[4], y, "audit", "Audit", ["notes on the plan;", "advisory, never blocks"], 3, "MUSE SPARK OPEN")]
    P += [p for p, _ in parts] + chain(y, dashed_into=(XS[4],))
    yc = max(e for _, e in parts) + 56
    P += [check(XS[1], yc, "allowlisted?", ["refused before", "any call"]),
          check(XS[2], yc, "on the tested menu?", ["a decline: the reason", "and the nearest", "tested option"]),
          check(XS[3], yc, "words kept, order right?", ["the build is blocked"])]
    # I, row 2: forecast, laptop, end screen, receipts
    y2 = yc + 150
    P.append(arrow(f"M{XS[3]},{yc + 44} V{yc + 76} H{XS[1]} V{y2 - 44}"))
    T_ = lambda x, yy, s: T(x, yy, 10.5, s, f'{MONO} letter-spacing="1.2" fill="{DIM}"')
    P.append(T_(XS[3] + 8, yc + 66, "YES"))
    parts2 = [node(XS[1], y2, "forecast", "Forecast", ["a recipe: story, condition", "and seed, by the jobs API"], 4, "TRIBE V2"),
              node(XS[2], y2, "laptop", "Laptop GPU", ["sandboxed worker, RTX 4050;", "w2v-BERT sound, Llama 3.2", "text in fp16, V-JEPA 2 off"]),
              node(XS[3], y2, "screen", "End screen", ["the gray forecast alone:", "Forecast, not a scan."]),
              node(XS[4], y2, "receipt", "Receipts and caps", ["a receipt for every call,", "no request text, no key;", "spend caps per provider,", "5 live runs an hour per visitor"])]
    P += [p for p, _ in parts2]
    P += [arrow(f"M{XS[1] + 50},{y2} L{XS[2] - 50},{y2}"), arrow(f"M{XS[2] + 50},{y2} L{XS[3] - 50},{y2}"),
          f'<path stroke="{DIM}" stroke-width="1" stroke-dasharray="2 4" d="M{XS[4] - 64},{y2} H{XS[3] + 64}"/>']
    yc2 = max(e for _, e in parts2[:3]) + 56
    P += [check(XS[1], yc2, "a saved forecast for this choice?", ["shown, labelled so; otherwise", "it runs live on the laptop,", "10:00-14:00 ET, Nov 4-9"], "YES")]
    # the ghost line, between the stations and the evidence
    yg = yc2 + 110
    P += [f'<path stroke="{LINE}" stroke-width="1" d="M46,{yg - 24} H1354"/>', ghost_band(ghost, yg)]
    # II. evidence
    y3s = yg + 220
    P.append(section(y3s, "II · EVIDENCE · OFFLINE, AGAINST PUBLIC SCANS"))
    y3 = y3s + 84
    parts3 = [node(XS[0], y3, "rules", "Rules first", ["PREREG.md (D0-D27), docs/rules:", "each pre-registered test’s rule", "before its brain data"]),
              node(XS[1], y3, "gpu", "Kaggle GPUs", ["forecasts of public stimuli,", "in private notebooks"], None, "TRIBE V2"),
              node(XS[2], y3, "scans", "Public scans", ["OpenNeuro: Le Petit Prince,", "Narratives, Emo-FilM"]),
              node(XS[3], y3, "evidence", "Scoring", ["each listener’s own brain, a", "shuffled null, cheap baselines"]),
              node(XS[4], y3, "results", "results/*.json", ["the files station 5 replays,", "each beside its baselines"], 5)]
    P += [p for p, _ in parts3] + chain(y3)
    yc3 = max(e for _, e in parts3) + 56
    P += [T(XS[3], yc3 - 12, 12.5, "scored once; pass, fail or not run,", f'{SERIF} font-style="italic" fill="{INK}" text-anchor="middle"'),
          T(XS[3], yc3 + 6, 12.5, "reported as it came", f'{SERIF} font-style="italic" fill="{INK}" text-anchor="middle"'),
          f'<path fill="none" stroke="{DIM}" stroke-width="1" stroke-dasharray="2 4" d="M{XS[0]},{yc3 - 30} V{yc3}"/>',
          T(XS[0], yc3 + 20, 12.5, "a git hook and a Muse Code", f'{SERIF} font-style="italic" fill="{DIM}" text-anchor="middle"'),
          T(XS[0], yc3 + 38, 12.5, "Stop hook block changes", f'{SERIF} font-style="italic" fill="{DIM}" text-anchor="middle"'),
          T(XS[0], yc3 + 56, 12.5, "to frozen code", f'{SERIF} font-style="italic" fill="{DIM}" text-anchor="middle"')]
    # III. film
    y4s = yc3 + 110
    P.append(section(y4s, "III · FILM · THE SAME LOOP ON VIDEO"))
    y4 = y4s + 84
    parts4 = [node(XS[0], y4, "reel", "Tears of Steel", ["release 0-190 s,", "CC BY 3.0, Blender Foundation"], 6),
              node(XS[1], y4, "mask", "Face masks", ["4,560 frames, through", "Meta’s Model API"], None, "SAM 3.1"),
              node(XS[2], y4, "film", "Fill and control", ["faces filled with the surrounding", "colour; an equal-area control", "inside TRIBE’s 470 x 470 view"]),
              node(XS[3], y4, "audit", "Edit check", ["Spark Open’s counts", "shown beside"], None, "MUSE GLIMMER 30B"),
              node(XS[4], y4, "scans", "Face test", ["forecasts of intact, filled", "and control; 28 viewers of", "the intact film, scored once"], None, "TRIBE V2")]
    P += [p for p, _ in parts4] + chain(y4)
    yc4 = max(e for _, e in parts4) + 56
    P += [check(XS[3], yc4, "did the edits work?", ["the film test", "does not run"]),
          check(XS[4], yc4, "gate 0: VIF at most 5?", ["NOT RUN,", "with its reason"])]
    # key and footer
    yk = yc4 + 120
    P += [f'<path stroke="{LINE}" stroke-width="1" d="M46,{yk - 30} H1354"/>', T(46, yk + 4, 11, "KEY", f'{MONO} letter-spacing="2" fill="{MUTE}"')]
    key = (("every run", f'<path fill="none" stroke="{INK}" stroke-width="1.3" marker-end="url(#ah)" d="M-16,0 h30"/>'),
           ("advisory, never blocks", f'<path fill="none" stroke="{INK}" stroke-width="1.3" stroke-dasharray="5 4" d="M-16,0 h30"/>'),
           ("a check: on no, the path stops and says why", f'<path fill="{BG}" stroke="{INK}" stroke-width="1.4" d="M0,-10 L10,0 L0,10 L-10,0 z"/>'),
           ("a Meta model", f'<rect x="-20" y="-9" width="40" height="17" fill="none" stroke="{DIM}"/>'),
           ("real listeners or viewers", f'<path fill="none" stroke="{AMB}" stroke-width="2" d="M-16,0 h30"/>'))
    for i, (lab, mark) in enumerate(key):
        x = 120 + (0, 180, 420, 790, 960)[i]
        P += [f'<g transform="translate({x},{yk})">{mark}</g>', T(x + 26, yk + 5, 13.5, lab, f'{SERIF} font-style="italic" fill="{MUTE}"')]
    P += [T(700, yk + 62, 13, "Keys stay in server environment variables. A hygiene scan runs before every push. TRIBE v2 is CC BY-NC 4.0 and is not included.",
            f'{SERIF} font-style="italic" fill="{DIM}" text-anchor="middle"'),
          T(700, yk + 92, 13, "Research use. Healthy volunteers and public data only. Not a diagnosis or medical advice.",
            f'{SERIF} font-style="italic" fill="{DIM}" text-anchor="middle"'), "</svg>"]
    svg = "".join(P)
    return svg.replace(f'viewBox="0 0 {W} {H}"', f'viewBox="0 0 {W} {yk + 130}"', 1).replace(f'height="{H}" fill="{BG}"', f'height="{yk + 130}" fill="{BG}"', 1) \
        .replace(crop_marks(W, H), crop_marks(W, yk + 130), 1)


def render(svg: str, out: Path, width: int):
    subprocess.run(["rsvg-convert", "-w", str(width), "-o", str(out)], input=svg.encode(), check=True)
    print(f"{out.relative_to(REPO)}: {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", type=Path, default=REPO.parent / "stillhere/atlas")
    ap.add_argument("--out", type=Path, default=REPO / "docs/assets")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ghost = json.loads((REPO / "results/ghostline_lpp_s8.json").read_text())
    hero = hero_svg(ghost, brain_png(a.atlas))
    render(hero, a.out / "stillhere-hero.png", 1800)
    render(hero, REPO / "web/public/hero.png", 1800)                       # the hosted page's header, the same picture
    render(architecture_svg(ghost), a.out / "stillhere-architecture.png", 1800)
