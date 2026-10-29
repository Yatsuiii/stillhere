"""Renders the 172 s v4 cut from video/scenes_v4.json (cloud plan v2, Oct 29 16:20 IST, items 8-9). Every scene carries a burned-in
headline (at most 7 words, 88 px at 1920 wide, so 49.5 px at 1080 wide) and the voice-over as burned-in subtitles; scenes start at
their planned second and crossfade into the next. A scene is a series of steps (the ghost line is built in the order the display rule
fixes: gray forecast, chip, coloured listeners, the thin 1-listener line) with at most one moving picture laid into its box: a terminal
recording re-drawn from its asciicast (real time; the planner's "Nearest validated option" sentence masked with a solid bar), Oct 29's
live run, or the film edit side by side. Ghost lines read results/ghostline_lpp_s8.json (docs/rules/display_rule_ghostline.md); until
that file exists a synthetic stand-in with the same layout is drawn under a PLACEHOLDER tag. Maps come from scripts/render_maps.py
(synthetic and tagged until results/face_map_RC.json exists). Slots (order, wall, d20, ec, film) take one ending each with --endings;
unfilled {placeholders} show as yellow tags on slides and as [name] in the subtitles. Also writes the subtitles (.srt) beside the video,
a contact sheet, and with --vo-md the voice-over script. Silent: the voice-over is recorded later.
usage: python scripts/render_video_v4.py [--endings slot=ending,...] [--values VALUES.json] [--out video/v4_placeholder.mp4] [--work DIR]
       python scripts/render_video_v4.py --vo-md docs/vo_script.md     (writes the script only)"""
import argparse
import html
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO)]
SCENES = REPO / "video/scenes_v4.json"
GHOST = REPO / "results/ghostline_lpp_s8.json"
FILM_GHOST = REPO / "results/ghostline_film.json"                         # the film ghost line (display rule, film section), Sunday
FILM = REPO.parent / "stillhere/pilot/film/film_stim_v2"                   # seg_intact.mp4, faces_removed_v2.mp4, area_matched_v2.mp4
RC_FILES = (REPO / "results/face_map_RC.json", REPO.parent / "stillhere-pilot-public/results/face_map_RC.json")
MENU = REPO / "stillhere/live_menu.json"                                 # the live menu; its screen summary fills {n_screened}
CACHE = REPO / "data/forecast_cache.json"                                # the page's saved runs (recomputed Oct 29 17:07 UTC), LIVE beat
GHOST_ORDER = ("forecast", "chip", "group", "five", "one")                # display rule draw order (group and five are "listeners")
BOX = (104, 176, 1712, 640)                                              # x, y, w, h of a scene's moving picture
E = html.escape
WORD = re.compile(r"\{[A-Za-z0-9_]+\}|[A-Za-z0-9$%~'][A-Za-z0-9$%~'.,:;/+-]*")       # a {placeholder} counts as one word


def words(text: str) -> int:
    return len(WORD.findall(text))


# ---- spec ----------------------------------------------------------------------------------------------------------------------
def resolve(sc: dict, endings: dict) -> list:
    """[(at, items)] steps of a scene under the chosen endings; a slot without an ending shows its default and a placeholder tag; None
    when the chosen ending leaves the scene out (D20 fail)."""
    steps = [(s["at"], s["items"]) for s in sc["steps"]] if "steps" in sc else [(0.0, sc.get("items", []))]
    if "slot" not in sc:
        return steps
    e = endings.get(sc["slot"])
    if e is None:
        extra = sc.get("default", []) + [{"kind": "placeholder", "text": f"{sc['slot']}: one of {', '.join(sc['variants'])}"}]
    elif sc["variants"][e] is None:
        return None
    else:
        extra = sc["variants"][e]
    at, items = steps[-1]                                                   # a slot's lines join the scene's last step
    return steps[:-1] + [(at, sc.get("common", []) + items + extra)]


def timeline(spec: dict, endings: dict) -> list:
    """(scene, start, end) of every scene shown; a scene an ending leaves out is covered by the scene before it, so the cut stays 172 s."""
    out = []
    for sc in spec["scenes"]:
        if resolve(sc, endings) is None:
            prev, a, _ = out[-1]; out[-1] = (prev, a, sc["end"])
        else:
            hv = sc.get("headline_variants", {}).get(endings.get(sc.get("slot")))
            out.append(({**sc, "headline": hv} if hv else sc, sc["start"], sc["end"]))
    return out


def vo_cues(spec: dict, endings: dict) -> list:
    out = []
    for c in spec["vo"]:
        text = c["text"]
        if "slot" in c:
            e = endings.get(c["slot"])
            text = c["variants"][e] if e else "{" + c["slot"] + "_vo}"
        out.append({"start": c["start"], "end": c["end"], "text": text})
    return out


def auto_values() -> dict:
    """Values read from committed files rather than typed in: the number of screened stories on the live menu."""
    return {"n_screened": json.loads(MENU.read_text())["screen_summary"]["n_on_menu"]}


def clip_title(clip_id: str) -> str:
    return next((c["short"] for c in json.loads(MENU.read_text())["clips"] if c["id"] == clip_id), clip_id)


def story_card_text(clip_id: str) -> str:
    """The story card of docs/rules/story_menu_rule.md section 6, word for word, from Fork A's formatter (web/stations.py story_card),
    so the page and the video cannot drift. When this interpreter lacks the web app's dependencies, the same function runs in the
    product environment."""
    try:
        from web import stations as ST
        return ST.story_card(clip_id)["text"]
    except ImportError:
        py = sorted(Path.home().glob(".local/share/uv/python/cpython-3.12*/bin/python3.12"))[0]
        site = next(p for v in (REPO, REPO.parent / "stillhere-app") for p in (v / ".venv/lib").glob("python3.12/site-packages"))
        code = f"import json; from web import stations as ST; print(json.dumps(ST.story_card({clip_id!r})['text']))"
        out = subprocess.run([str(py), "-c", code], cwd=REPO, env={**os.environ, "PYTHONPATH": f"{REPO}:{site}"}, capture_output=True,
                             text=True, check=True)
        return json.loads(out.stdout.strip().splitlines()[-1])


def fill(text: str, values: dict) -> str:
    out = E(text)
    for k, v in values.items():
        out = out.replace("{" + k + "}", E(str(v)))
    return re.sub(r"\{([A-Za-z0-9_]+)\}", lambda m: f'<span class="ph">{{{m.group(1)}}}</span>', out)


# ---- charts --------------------------------------------------------------------------------------------------------------------
def synthetic_ghost() -> dict:
    """Same layout as results/ghostline_lpp_s8.json (format stillhere-ghostline/1), made-up values, for placeholder frames only."""
    rng = np.random.default_rng(29); t = np.arange(10, 155) * 2.0
    def z(x): return ((x - x.mean()) / x.std(ddof=1)).round(3).tolist()
    base = np.convolve(rng.normal(size=t.size + 8), np.ones(8) / 8, "valid")[: t.size]
    lines = {"forecast": z(base), "group": z(base + 0.7 * rng.normal(size=t.size) * 0.3), "five": z(base + rng.normal(size=t.size) * 0.5),
             "one": z(base + rng.normal(size=t.size) * 1.1)}
    lab = {"forecast": "TRIBE v2 forecast", "group": "49 public listeners (average)", "five": "5 public listeners (average)",
           "one": "1 public listener"}
    return {"placeholder": True, "title": "Le Petit Prince, section 8 (LPP English)", "time_s": t.round(1).tolist(),
            "short_view": {"index": [0, 60]}, "chip": {"short": "Checked: held-out LPP",
                                                       "long": "held out; first measured Oct 22-23; LPP is one of TRIBE v2's own evaluation sets"},
            "lines": {k: {"label": lab[k], "values": v} for k, v in lines.items()},
            "frozen_line": {"text": "~5 volunteers' worth (at least 2). Group level."}}


def ghost_doc() -> dict:
    return json.loads(GHOST.read_text()) if GHOST.exists() else synthetic_ghost()


def _path(xs, ys, x0, y0, w, h, xr, yr) -> str:
    sx = lambda x: x0 + (x - xr[0]) / (xr[1] - xr[0]) * w
    sy = lambda y: y0 + h - (y - yr[0]) / (yr[1] - yr[0]) * h
    return "M" + " L".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))


STYLE = {"forecast": 'stroke="#8a8f98" stroke-width="5" stroke-dasharray="16 10" fill="none"',
         "group": 'stroke="#1f6fb4" stroke-width="5" fill="none"', "five": 'stroke="#d9822b" stroke-width="4" fill="none"',
         "one": 'stroke="#2a9d55" stroke-width="1.8" fill="none"'}


def ghost_svg(doc: dict, view: str, show: list, size: str = "full") -> str:
    """The ghost line as the display rule draws it; `show` is a prefix of the draw order, so a step can stop early but never reorder."""
    assert list(show) == list(GHOST_ORDER[: len(show)]), f"ghost steps must follow the display rule's draw order: {show}"
    t = np.array(doc["time_s"]); i0, i1 = doc["short_view"]["index"] if view == "short" else (0, len(t))
    W, H, x0, y0, w, h = {"full": (1712, 560, 70, 70, 1612, 400), "half": (820, 470, 30, 110, 770, 250),
                          "tile": (600, 420, 20, 104, 560, 190)}[size]
    half = size != "full"
    xs = t[i0:i1]; L = doc["lines"]; xr = (xs[0], xs[-1])
    top = max(abs(v) for k in ("forecast", "group", "five", "one") for v in L[k]["values"][i0:i1] if v is not None)
    yr = (-max(3.2, top + 0.2), max(3.2, top + 0.2))                       # one axis for every step; the values are never rescaled
    parts = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="Inter,Helvetica,Arial">',
             f'<text x="{x0}" y="36" font-size="28" fill="#1d1f23" font-weight="600">{E(doc["title"])}</text>',
             f'<line x1="{x0}" y1="{y0 + h / 2}" x2="{x0 + w}" y2="{y0 + h / 2}" stroke="#e3e0d8" stroke-width="2"/>',
             f'<text x="{x0}" y="{y0 + h + 34}" font-size="22" fill="#5d6470">seconds of the scan: {xs[0]:.0f} to {xs[-1]:.0f}</text>']
    shown = [k for k in ("forecast", "group", "five", "one") if k in show]
    for k in shown:
        parts.append(f'<path d="{_path(xs, L[k]["values"][i0:i1], x0, y0, w, h, xr, yr)}" {STYLE[k]}/>')
    per_row = {"full": 4, "half": 2, "tile": 1}[size]
    for n, k in enumerate(shown):
        lx = x0 + (n % per_row) * (w / per_row); ly = y0 + h + 70 + (n // per_row) * 34
        parts.append(f'<line x1="{lx:.0f}" y1="{ly - 8}" x2="{lx + 44:.0f}" y2="{ly - 8}" {STYLE[k]}/>'
                     f'<text x="{lx + 54:.0f}" y="{ly}" font-size="22" fill="#1d1f23">{E(L[k]["label"])}</text>')
    if "chip" in show:
        c = doc["chip"]["short"]; cw = 15 * len(c); cy = 52 if half else 12
        parts.append(f'<rect x="{W - 12 - cw}" y="{cy}" rx="20" width="{cw}" height="40" fill="#eef4ea" stroke="#2a7a3b" stroke-width="2"/>'
                     f'<text x="{W - 12 - cw / 2}" y="{cy + 28}" font-size="24" text-anchor="middle" fill="#1e5a2b">{E(c)}</text>')
    if doc.get("placeholder"):
        parts.append(f'<text x="{x0 + w / 2}" y="{y0 + h / 2 + 14}" font-size="{30 if half else 40}" text-anchor="middle" fill="#6b5600" '
                     'opacity="0.8">PLACEHOLDER: synthetic lines</text>')
    return "".join(parts) + "</svg>"


def frozen_html(doc: dict) -> str:
    """The frozen section-8 line; drawn only beside the LPP ghost line (display rule, Numbers on screen)."""
    return f'<div class="frozen">{E(doc["frozen_line"]["text"])}</div>'


def curve_svg(show: list) -> str:
    """The volunteer curve from results/section8.json in ghost style: forecast level gray dashed first, real volunteers in colour."""
    s = json.loads((REPO / "results/section8.json").read_text()); cur, ve = s["isc_curve_language"], s["volunteer_equivalent_language"]
    W, H = 820, 470; x0, y0, w, h = 80, 70, 700, 320; xr, yr = (0, 16), (0, 2.6)
    sx = lambda x: x0 + x / xr[1] * w; sy = lambda y: y0 + h - y / yr[1] * h
    p = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="Inter,Helvetica,Arial">',
         f'<text x="{x0}" y="36" font-size="28" fill="#1d1f23" font-weight="600">Real volunteers needed to match it</text>',
         f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" stroke="#9aa3ad" stroke-width="2"/>',
         f'<text x="{x0 + w / 2}" y="{y0 + h + 50}" font-size="22" fill="#5d6470" text-anchor="middle">volunteers averaged (language areas)</text>']
    for k in (1, 5, 10, 15):
        p.append(f'<text x="{sx(k):.0f}" y="{y0 + h + 26}" font-size="20" fill="#5d6470" text-anchor="middle">{k}</text>')
    if "forecast" in show:
        p.append(f'<line x1="{x0}" y1="{sy(ve["z"]):.1f}" x2="{x0 + w}" y2="{sy(ve["z"]):.1f}" {STYLE["forecast"]}/>'
                 f'<text x="{x0 + w}" y="{sy(ve["z"]) - 12:.1f}" font-size="22" fill="#5d6470" text-anchor="end">TRIBE v2 forecast</text>')
    if "real" in show:
        p.append('<path d="M' + " L".join(f"{sx(c['k']):.1f},{sy(c['z']):.1f}" for c in cur) + f'" {STYLE["group"]}/>')
        p += [f'<circle cx="{sx(c["k"]):.1f}" cy="{sy(c["z"]):.1f}" r="7" fill="#1f6fb4"/>' for c in cur]
    if "marks" in show:
        for k, lab in ((ve["point_k"], "about 5"), (ve["lower_k"], "at least 2")):
            p.append(f'<line x1="{sx(k):.1f}" y1="{y0}" x2="{sx(k):.1f}" y2="{y0 + h}" stroke="#d9822b" stroke-width="3" stroke-dasharray="6 6"/>'
                     f'<text x="{sx(k) + 8:.1f}" y="{y0 + 24}" font-size="22" fill="#9a5a14">{lab}</text>')
    return "".join(p) + "</svg>"


def cache_entry(spec: dict, which: str) -> dict:
    """The saved run behind a LIVE panel: spec["live"][which] names its clip and condition (default seed)."""
    want = spec["live"][which]; d = json.loads(CACHE.read_text())
    return next(e for e in d["entries"].values() if all(e["recipe"][k] == v for k, v in want.items()))


COND = {"intact": "intact", "sentence_scramble": "sentences shuffled", "word_scramble": "words shuffled", "reversed": "reversed"}


def livetc_svg(e: dict, W: int = 1712, H: int = 330) -> str:
    """One saved run (data/forecast_cache.json), gray only: language and hearing areas side by side, 1 s samples."""
    out = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="Inter,Helvetica,Arial">']
    pw = W / 2; n = e["seconds"]
    for k, (reg, name) in enumerate((("language", "Language areas"), ("hearing", "Hearing areas"))):
        x0, y0, w, h = k * pw + 20, 44, pw - 60, H - (110 if W >= 1400 else 130)
        vals = e["regions"][reg]; yr = (min(vals) - 0.05, max(vals) + 0.05)
        out.append(f'<text x="{x0}" y="28" font-size="26" fill="#1d1f23" font-weight="600">{name}</text>'
                   f'<path d="{_path(np.arange(len(vals)) * e["tr_s"], vals, x0, y0, w, h, (0, n), yr)}" {STYLE["forecast"]}/>'
                   f'<text x="{x0}" y="{y0 + h + 30}" font-size="20" fill="#5d6470">0 to {n} s of the clip</text>')
    r = e["recipe"]; ch = clip_title(r["clip_id"])
    seed = f", seed {r['seed']}" if r["seed"] else ""
    first, second = f"TRIBE v2 forecast, {COND[r['condition']]}{seed}.", f"LibriVox, {ch}, {n} s. Saved run. Forecast, not a scan."
    if W >= 1400:
        out.append(f'<text x="20" y="{H - 12}" font-size="22" fill="#1d1f23">{E(first)} {E(second)}</text>')
    else:                                                                   # a narrow panel: two lines, so nothing is clipped
        out.append(f'<text x="20" y="{H - 36}" font-size="22" fill="#1d1f23">{E(first)}</text>'
                   f'<text x="20" y="{H - 10}" font-size="22" fill="#1d1f23">{E(second)}</text>')
    return "".join(out) + "</svg>"


def lanes_html(e: dict) -> str:
    """TRIBE's three input lanes for a saved run, as data/forecast_cache.json records them."""
    m = e["models"]; n = e["n_words_text_lane"]
    assert m["v_jepa_2"].startswith("off")
    rows = [("w2v-BERT: sound", "on", 100), ("Llama 3.2: words in context", f"{n} words", 100 if n else 0), ("V-JEPA 2: off (audio)", "off", 0)]
    return '<div class="lanes">' + "".join(
        f'<div class="lane"><div class="bar"><div class="fill" style="width:{pct}%"></div></div><span>{E(a)}</span>'
        f'<span class="st">{E(st)}</span></div>' for a, st, pct in rows) + "</div>"


def receipt_text(e: dict) -> str:
    t = e["timings_s"]
    return (f"Saved run, {e['computed_utc'][:10]} {e['computed_utc'][11:16]} UTC: total {t['total']:.1f} s (text lane {t['text_lane']:.1f} s, "
            f"audio lane {t['audio_lane']:.1f} s, predict {t['predict']:.1f} s).")


def face_ticks() -> list:
    """[film second, share of the 1280 x 534 frame covered by faces] for every second from 0 to 185 s with a share above 0: the face
    rule's F before any HRF (edits_v2.json field px), as the film section of the display rule draws it. Stimulus file only."""
    pf = json.loads((FILM / "edits_v2.json").read_text())["per_frame"]
    share = np.array([sum(f["px"] for f in fr["faces"]) for fr in pf], float) / (1280 * 534)
    per_s = share[: len(share) // 24 * 24].reshape(-1, 24).mean(1)
    return [[s, round(float(v), 4)] for s, v in enumerate(per_s[:186]) if v > 0]


def synthetic_film_ghost() -> dict:
    """Same layout as results/ghostline_film.json (time_s, lines, chip, ticks), made-up lines on the rule's grid (film time 10 to 185 s,
    135 points at 1.3 s); the ticks are real. For placeholder frames only."""
    rng = np.random.default_rng(31); t = np.arange(10, 185 + 1e-9, 1.3)[:135]
    def z(x): return ((x - x.mean()) / x.std(ddof=1)).round(3).tolist()
    base = np.convolve(rng.normal(size=t.size + 5), np.ones(5) / 5, "valid")[: t.size]
    lab = {"forecast": "TRIBE v2 forecast", "group": "{N} public viewers (average)", "one": "1 public viewer"}
    vals = {"forecast": z(base), "group": z(base + rng.normal(size=t.size) * 0.5), "one": z(base + rng.normal(size=t.size) * 1.0)}
    return {"placeholder": True, "time_s": t.round(1).tolist(), "lines": {k: {"label": lab[k], "values": v} for k, v in vals.items()},
            "chip": {"short": "Film: rule {film_commit}", "long": "rule committed before any viewer's scan was read · {film_commit}"},
            "ticks": face_ticks() if (FILM / "edits_v2.json").exists() else []}


def film_ghost_doc() -> dict:
    return json.loads(FILM_GHOST.read_text()) if FILM_GHOST.exists() else synthetic_film_ghost()


def filmghost_svg(doc: dict, show: list, title_html: str, values: dict) -> str:
    """The film ghost line (display rule, film section): face ticks and the gray dashed forecast first, then the group line, then the
    thin single viewer, on film time 0 to 185 s; gaps stay gaps. The chip is a separate item, drawn from the second step on."""
    assert list(show) == ["forecast", "group", "one"][: len(show)], show
    W, H = 1712, 470; x0, y0, w, h = 70, 20, 1600, 330; xr, yr = (0, 185), (-3.2, 3.2)
    sx = lambda x: x0 + (x - xr[0]) / (xr[1] - xr[0]) * w
    sy = lambda y: y0 + h - (y - yr[0]) / (yr[1] - yr[0]) * h
    p = [f'<div class="charttitle">{title_html}</div>',
         f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="Inter,Helvetica,Arial">']
    for s_, v in doc.get("ticks") or []:
        p.append(f'<line x1="{sx(s_):.1f}" y1="{y0 + h + 40}" x2="{sx(s_):.1f}" y2="{y0 + h + 40 - 30 * min(1, v / 0.1):.1f}" stroke="#7a4fb0" stroke-width="5"/>')
    t = doc["time_s"]
    for k in show:
        segs, cur = [], []
        for x, y in zip(t, doc["lines"][k]["values"]):
            if y is None:
                segs.append(cur); cur = []
            else:
                cur.append(f"{sx(x):.1f},{sy(y):.1f}")
        for seg in segs + [cur]:
            if len(seg) > 1:
                p.append(f'<path d="M{" L".join(seg)}" {STYLE[k]}/>')
    lx = x0
    for k in show:
        lab = fill(doc["lines"][k]["label"], values)
        p.append(f'<line x1="{lx}" y1="{H - 14}" x2="{lx + 44}" y2="{H - 14}" {STYLE[k]}/><text x="{lx + 54}" y="{H - 6}" font-size="22" fill="#1d1f23">'
                 f'{re.sub(r"<[^>]+>", "", lab)}</text>')
        lx += 120 + 11 * len(doc["lines"][k]["label"])
    p.append(f'<text x="{x0}" y="{y0 + h + 66}" font-size="22" fill="#5d6470">Ticks: share of the frame covered by faces, each second (film time 0 to 185 s)</text>')
    if doc.get("placeholder"):
        p.append(f'<text x="{W / 2}" y="{y0 + h / 2}" font-size="40" text-anchor="middle" fill="#6b5600" opacity="0.8">PLACEHOLDER: synthetic lines (Sunday)</text>')
    return "".join(p) + "</svg>"


def endscreen_html(pair: str, values: dict, ctx: dict) -> str:
    """The LIVE beat's last screen: the laptop forecast alone in gray, and beside it, as a separate tile, real listeners on another story:
    notthefall on a D20 pass ("Real listeners, different story."), the LPP ghost line on a fail ("Real listeners, held-out story")."""
    left = ('<div class="chip">Forecast only · audio-only, laptop</div>' + livetc_svg(ctx["live"]["live"], 1060, 350)
            + '<div class="small">No public scan of this clip. Forecast, not a scan.</div>')
    if pair == "d20":
        right = ('<div class="chip">notthefall: rule 340b74b</div><div class="tiletitle">Real listeners, different story.</div>'
                 f'<div class="small">{fill(ctx["d20_line"], values)}</div>')
    elif pair == "lpp":
        right = ghost_svg(ctx["ghost"], "short", ["forecast", "chip", "group"], "tile") + '<div class="tiletitle">Real listeners, held-out story</div>'
    else:
        right = '<div class="phbox">PLACEHOLDER: d20 pass (notthefall tile) or fail (LPP ghost line), tonight</div>'
    return f'<div class="row"><div class="col">{left}</div><div class="tile">{right}</div></div>'


def people_svg(n: int) -> str:
    one = ('<svg width="96" height="150" viewBox="0 0 96 150" xmlns="http://www.w3.org/2000/svg"><circle cx="48" cy="34" r="24" fill="#1f4e79"/>'
           '<path d="M10 146 Q10 70 48 70 Q86 70 86 146 Z" fill="#1f4e79"/></svg>')
    return '<div class="people">' + one * n + "</div>"


# ---- slides --------------------------------------------------------------------------------------------------------------------
CSS = """
html,body{margin:0;width:1920px;height:1080px;overflow:hidden;background:#fbfaf7;color:#1d1f23;font-family:Inter,"Noto Sans",Helvetica,Arial,sans-serif}
main{position:relative;width:1920px;height:1080px}
h1{position:absolute;left:104px;top:34px;margin:0;font-size:88px;line-height:1.05;font-weight:800;letter-spacing:-1px;white-space:nowrap;color:#14171c}
.content{position:absolute;left:104px;top:176px;width:1712px;height:704px;display:flex;flex-direction:column;gap:22px}
.row{display:flex;gap:72px;align-items:flex-start}
.title{font-size:40px;font-weight:600;color:#1f4e79}
.text{font-size:38px;line-height:1.35} .small{font-size:27px;line-height:1.4;color:#5d6470} .big{font-size:54px;font-weight:700;line-height:1.2;color:#1f4e79}
.frozen{font-size:40px;font-weight:700;color:#1f6fb4;margin-top:8px}
.chip{align-self:flex-start;font-size:30px;padding:8px 22px;border-radius:30px;background:#eef4ea;border:2px solid #2a7a3b;color:#1e5a2b}
table{font-size:34px;border-collapse:collapse} td{padding:10px 40px 10px 0;vertical-align:top} td:first-child{font-weight:600;color:#1f4e79}
.people{display:flex;gap:14px;flex-wrap:nowrap}
.alarm{font-size:46px;font-weight:700;color:#fff;background:#b42318;padding:34px 40px;border-radius:16px;line-height:1.3}
.adv{border:3px solid #9aa3ad;border-radius:16px;padding:26px 32px;font-size:36px;background:#fff}
.adv .lab{display:inline-block;font-size:26px;padding:4px 16px;border-radius:20px;background:#eceef1;color:#5d6470;margin-bottom:12px}
.kara{font-size:34px;line-height:1.5;background:#fff;border-radius:14px;padding:20px 28px;border:2px solid #e3e0d8;color:#5d6470}
.kara b{color:#1d1f23}
.check{font-size:36px} .check td{padding:8px 30px 8px 0} .ok{color:#2a7a3b;font-weight:800} .wait{color:#9aa3ad;font-weight:800}
.lanes{display:flex;flex-direction:column;gap:18px}
.lane{display:flex;align-items:center;gap:24px;font-size:36px} .lane .bar{width:520px;height:34px;border-radius:17px;background:#e3e0d8;overflow:hidden}
.lane .fill{height:100%;background:#1f6fb4} .lane .st{font-weight:700;color:#5d6470}
.maps img{width:1712px;display:block;margin-bottom:8px}
.stamp{align-self:flex-start;font-size:110px;font-weight:900;letter-spacing:6px;padding:6px 40px;border:10px solid;border-radius:18px;transform:rotate(-4deg);margin:18px 0}
.stamp.pass{color:#2a7a3b;border-color:#2a7a3b} .stamp.fail{color:#b42318;border-color:#b42318} .stamp.neutral{color:#5d6470;border-color:#5d6470}
.rail{display:grid;grid-template-columns:520px 1fr;gap:16px 36px;font-size:38px} .rail .m{font-weight:800;color:#1f4e79}
.box{position:absolute;left:104px;top:176px;width:1712px;height:640px;background:#11141a;border-radius:10px}
.boxlab{position:absolute;left:104px;top:826px;font-size:24px;color:#5d6470}
.film3{position:absolute;left:104px;top:176px;width:1712px;height:230px;background:#11141a}
.film3lab{position:absolute;top:414px;font-size:26px;color:#1d1f23;width:552px;text-align:center}
.below{position:absolute;left:104px;top:470px;width:1712px;display:flex;flex-direction:column;gap:18px}
.ph{background:#ffe58a;color:#4a3b00;padding:0 6px;border-radius:5px;font-weight:600}
.phbox{border:3px dashed #d4a500;padding:12px 18px;font-size:26px;color:#6b5600;border-radius:10px;background:#fff8d6}
.col{width:1060px;display:flex;flex-direction:column;gap:12px} .tile{width:600px;border:2px solid #e3e0d8;border-radius:14px;padding:14px;background:#fff;display:flex;flex-direction:column;gap:10px}
.tiletitle{font-size:34px;font-weight:700;color:#1f6fb4} .charttitle{font-size:30px;font-weight:600}
.maps2{display:flex;gap:32px} .maps2 figure{margin:0;width:840px} .maps2 img{width:840px;display:block} .maptitle{font-size:28px;font-weight:600;margin-bottom:6px}
.tag2{align-self:flex-start;font-size:26px;padding:4px 16px;border-radius:20px;background:#eceef1;color:#5d6470;font-weight:600}
.cutnote{align-self:flex-start;font-size:34px;font-weight:700;color:#fff;background:#11141a;padding:8px 20px;border-radius:10px}
.storycard{font-size:36px;line-height:1.45;background:#fff;border:2px solid #e3e0d8;border-radius:14px;padding:24px 30px}
.receipt{font-size:24px;color:#5d6470;font-family:"Noto Sans Mono",monospace}
.foot{position:absolute;left:104px;right:104px;bottom:12px;font-size:18px;color:#5d6470;line-height:1.3}
"""


def item_html(it: dict, values: dict, ctx: dict) -> str:
    k = it["kind"]
    if k in ("title", "text", "small", "big"):
        return f'<div class="{k}">{fill(it["text"], values)}</div>'
    if k == "chip":
        return f'<div class="chip">{fill(it["text"], values)}</div>'
    if k == "card":
        return "<table>" + "".join("<tr>" + "".join(f"<td>{fill(c, values)}</td>" for c in r) + "</tr>" for r in it["rows"]) + "</table>"
    if k == "people":
        return people_svg(it["n"])
    if k == "alarm":
        return f'<div class="alarm">{fill(it["text"], values)}</div>'
    if k == "advisory":
        return f'<div class="adv"><span class="lab">{E(it["label"])}</span><br>{fill(it["text"], values)}</div>'
    if k == "placeholder":
        return f'<div class="phbox">PLACEHOLDER: {E(it["text"])}</div>'
    if k == "karaoke":
        m = json.loads((REPO / "results/conditions_manipulation_check.json").read_text())["conditions"]["sentence_scramble"]
        w = m["transcript"].split()[:46]
        return ('<div class="small">Voice Transcribe, reading the built sentence scramble back:</div>'
                f'<div class="kara"><b>{E(" ".join(w[:14]))}</b> {E(" ".join(w[14:]))} ...</div>'
                '<div class="phbox">PLACEHOLDER: word-by-word highlight from Voice Transcribe word times (Friday)</div>')
    if k == "checklist":
        m = json.loads((REPO / "results/conditions_manipulation_check.json").read_text())["conditions"]
        ss, rv = m["sentence_scramble"], m["reversed"]
        rows = [("Every sentence present once", "{sentences_once}"),
                ("Words heard, of the original", f"{float(ss['recall']):.0%}"),
                ("Order changed", f"matches the new order {float(ss['order_vs_expected']):.0%}, the original {float(ss['order_vs_original']):.0%}"),
                ("Reversed: words heard", str(int(rv["n_words_heard"])))]
        done = it["state"] == "done"
        mark = '<span class="ok">&#10003;</span>' if done else '<span class="wait">&#8230;</span>'
        return ('<table class="check">' + "".join(f"<tr><td>{mark}</td><td>{E(a)}</td><td>{fill(b, values) if done else ''}</td></tr>"
                                                    for a, b in rows) + "</table>")
    if k == "ghost":
        out = ghost_svg(ctx["ghost"], it["view"], it["show"], "half" if it.get("half") else "full")
        assert "five" not in it["show"] or it.get("frozen_line"), "the 5-listener line only where the about-5 line is shown"
        return out + (frozen_html(ctx["ghost"]) if it.get("frozen_line") and not it.get("half") else "")
    if k == "curve":
        return curve_svg(it["show"])
    if k == "livetc":
        return livetc_svg(ctx["live"][it["entry"]])
    if k == "lanes":
        return lanes_html(ctx["live"][it["entry"]])
    if k == "receipt":
        return f'<div class="receipt">{E(receipt_text(ctx["live"][it["entry"]]))}</div>'
    if k == "endscreen":
        return endscreen_html(it["pair"], values, ctx)
    if k == "filmghost":
        return filmghost_svg(ctx["film_ghost"], it["show"], fill(it["title"], values), values)
    if k == "maps":
        return '<div class="maps2">' + "".join(f'<figure><div class="maptitle">{fill(m["title"], values)}</div>'
                                               f'<img src="{ctx["maps"][m["key"]].as_uri()}"></figure>' for m in it["maps"]) + "</div>"
    if k == "storycard":
        return f'<div class="storycard">{E(ctx["cards"][it["clip"]])}</div>'
    if k == "tag":
        return f'<div class="tag2">{E(it["text"])}</div>'
    if k == "cutnote":
        return f'<div class="cutnote">{fill(it["text"], values)}</div>'
    if k == "take":
        return '<div class="box"></div>' if values.get("live_take") else \
            '<div class="phbox">PLACEHOLDER: the filmed take, cut once (values live_take, run_s and n from its receipt)</div>'
    if k == "stamp":
        return f'<div class="stamp {it["tone"]}">{E(it["text"])}</div>'
    if k == "rail":
        return '<div class="rail">' + "".join(f'<div class="m">{E(a)}</div><div>{E(b)}</div>' for a, b in it["rows"]) + "</div>"
    if k == "cast":
        return f'<div class="box"></div><div class="boxlab">{E(it["label"])}</div>'
    if k == "film3":
        return ('<div class="film3"></div>' + "".join(f'<div class="film3lab" style="left:{104 + i * 580}px">{lab}</div>'
                                                      for i, lab in enumerate(("Intact", "Faces filled", "Same-size control"))))
    raise ValueError(k)


MOVING = ("cast", "film3", "take")


def scene_html(sc: dict, items: list, values: dict, spec: dict, ctx: dict) -> str:
    moving = [i for i in items if i["kind"] in MOVING and (i["kind"] != "take" or values.get("live_take"))]
    rest = [i for i in items if i not in moving]
    if any(i["kind"] == "take" for i in moving):                       # cut mode with the filmed take: the take and its cut note only
        rest = [i for i in rest if i["kind"] == "cutnote"]
    body = "".join(item_html(i, values, ctx) for i in moving)
    inner = "".join(item_html(i, values, ctx) for i in rest)
    if moving and moving[0]["kind"] == "film3":
        body += f'<div class="below">{inner}</div>'
    elif moving:
        body += f'<div class="below" style="top:860px">{inner}</div>' if inner else ""
    else:
        ghost = [i for i in rest if i["kind"] in ("ghost", "curve")]
        if len(ghost) == 2:                                               # ghost line and volunteer curve side by side
            others = [i for i in rest if i not in ghost]
            fz = frozen_html(ctx["ghost"]) if any(i.get("frozen_line") for i in ghost) else ""
            inner = ('<div class="row">' + "".join(f"<div>{item_html(i, values, ctx)}</div>" for i in ghost) + "</div>" + fz
                     + "".join(item_html(i, values, ctx) for i in others))
        body += f'<div class="content">{inner}</div>'
    foot = []
    if sc.get("research"):
        foot.append(spec["research"])
    if any(i["kind"] == "ghost" or (i["kind"] == "endscreen" and i["pair"] == "lpp") for i in items):
        foot.append("Checked: held-out LPP: " + ctx["ghost"]["chip"]["long"] + ".")
    if any(i["kind"] == "chip" and i["text"].startswith("Film: rule") for i in items):
        foot.append("Film: rule {film_commit}: rule committed before any viewer's scan was read · {film_commit}.")
    if sc.get("film"):
        foot.append(spec["credit"])
    if sc.get("results"):
        foot.append(spec["nd"])
    f = f'<div class="foot">{fill("  ".join(foot), values)}</div>' if foot else ""
    return (f'<!doctype html><meta charset="utf-8"><style>{CSS}</style><main><h1>{fill(sc["headline"], values)}</h1>'
            f"{body}{f}</main>")


# ---- moving pictures -----------------------------------------------------------------------------------------------------------
def mask_text(text: str, pattern: str) -> str:
    """Replaces every match with full blocks of the same length, so the line is visibly masked and nothing else moves."""
    return re.sub(pattern, lambda m: "".join(ch if ch in "\r\n" else "█" for ch in m.group(0)), text)


def cast_clip(it: dict, seconds: float, out: Path) -> None:
    import render_cast as RC
    from PIL import Image, ImageDraw, ImageFont
    W, H, FPS, SIZE, LINE = BOX[2], BOX[3], 10, 26, 33
    font, bold = ImageFont.truetype(RC.FONT, SIZE), ImageFont.truetype(RC.BOLD, SIZE)
    ev = RC.events(str(REPO / it["file"]))
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-vf", "fps=30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(out)],
                          stdin=subprocess.PIPE)
    for n in range(int(seconds * FPS) + 1):
        t = it["t0"] + min(n / FPS, it["t1"] - it["t0"])
        text = "".join(d for s, d in ev if s <= t)
        if it.get("mask"):
            text = mask_text(text, it["mask"])
        rows = RC.screen(text)[-it["rows"]:]
        im = Image.new("RGB", (W, H), RC.BG); d = ImageDraw.Draw(im)
        d.rectangle([0, 0, W, 50], fill=RC.BAR)
        d.text((30, 11), "StillHere Pilot, live run (demo, not scored)", font=bold, fill=RC.FG)
        clock = f"real time, not sped up   {int(t) // 60:02d}:{int(t) % 60:02d}"
        d.text((W - 30 - d.textlength(clock, font=font), 11), clock, font=font, fill=RC.DIM)
        y = 66
        for r, style in RC.styled(rows):
            d.text((30, y), r, font=bold if style in (RC.REQ, RC.ACC, RC.DEC) else font, fill=style); y += LINE
        ff.stdin.write(im.tobytes())
    ff.stdin.close()
    assert ff.wait() == 0, "ffmpeg failed on a cast clip"


def moving_clip(it: dict, seconds: float, work: Path, idx: int, values: dict = None):
    """(path, x, y) of the scene's moving picture, already sized for its box and at least `seconds` long."""
    out = work / f"moving_{idx:02d}.mp4"
    if it["kind"] == "cast":
        cast_clip(it, seconds, out); return out, BOX[0], BOX[1]
    if it["kind"] == "take":                                                # cut mode: the take's start, one cut, then its end
        take = (values or {}).get("live_take")
        if not take:
            return None
        w, h, head = BOX[2], BOX[3], it["keep_head_s"]
        vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x11141a,fps=30,setsar=1"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-t", f"{head}", "-i", take, "-sseof", f"-{seconds - head:.3f}", "-i", take,
                        "-filter_complex", f"[0:v]{vf}[a];[1:v]{vf}[b];[a][b]concat=n=2:v=1:a=0[o]", "-map", "[o]", "-an", "-c:v", "libx264",
                        "-crf", "18", "-pix_fmt", "yuv420p", str(out)], check=True)
        return out, BOX[0], BOX[1]
    if it["kind"] == "film3":
        tw, th, gap = 552, 230, 28
        args = ["ffmpeg", "-y", "-loglevel", "error"]
        for f in ("seg_intact.mp4", "faces_removed_v2.mp4", "area_matched_v2.mp4"):
            args += ["-ss", str(it["t0"]), "-t", f"{seconds:.3f}", "-i", str(FILM / f)]
        fc = ";".join(f"[{i}:v]scale={tw}:{th},fps=30,setsar=1[v{i}]" for i in range(3))
        fc += f";color=c=0x11141a:s={3 * tw + 2 * gap}x{th}:r=30:d={seconds:.3f}[bg]"
        fc += f";[bg][v0]overlay=0:0[a];[a][v1]overlay={tw + gap}:0[b];[b][v2]overlay={2 * (tw + gap)}:0,tpad=stop_mode=clone:stop_duration={seconds}[o]"
        subprocess.run(args + ["-filter_complex", fc, "-map", "[o]", "-t", f"{seconds:.3f}", "-an", "-c:v", "libx264", "-crf", "16",
                               "-pix_fmt", "yuv420p", str(out)], check=True)
        return out, 104, 176
    raise ValueError(it["kind"])


def inset_clip(spec: dict, values: dict, work: Path):
    """(path, seconds) of the corner inset: the filmed live job (values live_take) uncut, or a placeholder of the same size with an
    elapsed clock, run_s seconds long (values run_s from the take's receipt, timings_s.total; the spec's placeholder length otherwise)."""
    ins = spec["live"]["inset"]; run_s = float(values.get("run_s") or ins["placeholder_s"])
    take = values.get("live_take")
    if take:
        return Path(take), run_s
    from PIL import Image, ImageDraw, ImageFont
    import render_cast as RC
    W, H, FPS = ins["w"], ins["h"], 10; out = work / "inset.mp4"
    big, small = ImageFont.truetype(RC.BOLD, 40), ImageFont.truetype(RC.FONT, 15)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-vf", "fps=30", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for k in range(int(run_s * FPS) + 1):
        im = Image.new("RGB", (W, H), (17, 20, 24)); d = ImageDraw.Draw(im); t = k / FPS
        d.rectangle([0, 0, W - 1, H - 1], outline=(212, 165, 0), width=4)
        d.text((12, 10), "PLACEHOLDER: Sunday's live job, uncut", font=small, fill=(255, 229, 138))
        d.text((16, H / 2 - 30), f"{int(t) // 60:02d}:{int(t) % 60:02d}", font=big, fill=(214, 218, 224))
        d.text((16, H - 34), f"until it lands ({run_s:.0f} s)", font=small, fill=(128, 136, 148))
        ff.stdin.write(im.tobytes())
    ff.stdin.close()
    assert ff.wait() == 0, "ffmpeg failed on the inset"
    return out, run_s


def scene_clip(pngs: list, ats: list, seconds: float, moving, out: Path, fps: int) -> None:
    durs = [b - a for a, b in zip(ats, ats[1:] + [seconds])]
    args = ["ffmpeg", "-y", "-loglevel", "error"]
    for p, d in zip(pngs, durs):
        args += ["-loop", "1", "-framerate", str(fps), "-t", f"{d:.3f}", "-i", str(p)]
    n = len(pngs)
    fc = "".join(f"[{i}:v]fps={fps},format=yuv420p,setsar=1[s{i}];" for i in range(n))
    fc += "".join(f"[s{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[bg]"
    last = "bg"
    if moving:
        path, x, y = moving
        args += ["-i", str(path)]
        fc += f";[{n}:v]fps={fps},format=yuv420p,setsar=1[mv];[bg][mv]overlay={x}:{y}:eof_action=repeat[o]"; last = "o"
    args += ["-filter_complex", fc, "-map", f"[{last}]", "-t", f"{seconds:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "16", "-pix_fmt", "yuv420p", "-r", str(fps), str(out)]
    subprocess.run(args, check=True)


# ---- subtitles and the voice-over script ---------------------------------------------------------------------------------------
def _ts(s: float, srt: bool) -> str:
    h, m, sec = int(s // 3600), int(s % 3600 // 60), s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",") if srt else f"{h}:{m:02d}:{sec:05.2f}"


def sub_text(text: str, values: dict, ass: bool) -> str:
    for k, v in values.items():
        text = text.replace("{" + k + "}", str(v))
    if ass:
        return re.sub(r"\{([A-Za-z0-9_]+)\}", r"{\\c&H00D7FF&}[\1]{\\r}", text)
    return re.sub(r"\{([A-Za-z0-9_]+)\}", r"[\1]", text)


def write_subs(cues: list, values: dict, spec: dict, ass_path: Path, srt_path: Path) -> None:
    W, H = spec["size"]
    ass = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 0", "",
           "[V4+ Styles]", "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
           "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
           f"Style: VO,Noto Sans,{spec['subtitle_px']},&H00FFFFFF,&H00FFFFFF,&H28181410,&H28181410,0,0,0,0,100,100,0,0,3,12,0,2,220,220,96,1",
           "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    srt = []
    for i, c in enumerate(cues, 1):
        ass.append(f"Dialogue: 0,{_ts(c['start'], False)},{_ts(c['end'], False)},VO,,0,0,0,,{sub_text(c['text'], values, True)}")
        srt += [str(i), f"{_ts(c['start'], True)} --> {_ts(c['end'], True)}", sub_text(c["text"], values, False), ""]
    ass_path.write_text("\n".join(ass) + "\n"); srt_path.write_text("\n".join(srt))


def owner(spec: dict, cue: dict) -> dict:
    """The scene a voice-over cue starts in (cues may run on into the next scene)."""
    return next(sc for sc in spec["scenes"] if sc["start"] <= cue["start"] + 0.3 < sc["end"])


def vo_markdown(spec: dict) -> str:
    """The voice-over script for the recording session: every cue with its time, length, word count and rate, scene by scene."""
    lines = ["# Voice-over script, v4 cut (172 s)", "",
             "Generated from video/scenes_v4.json by `python scripts/render_video_v4.py --vo-md docs/vo_script.md`; do not edit by hand.",
             "Read at about 2.5 words a second. The first four lines are the frozen D23 opener, word for word; they start under the cold",
             "open. Lines in {braces} are filled before recording. The d20 and film result lines have one reading per ending; record the",
             "one that happened. On a d20 fail the notthefall scene leaves and the story wall runs on to 118 s. Yatsu records Monday.", ""]
    total_words = 0; auto = auto_values()
    for sc in spec["scenes"]:
        cues = [c for c in spec["vo"] if owner(spec, c) is sc]
        lines += [f"## {sc['start']:g}-{sc['end']:g} s · {sc['id']} · headline: \"{sc['headline']}\"", ""]
        if sc.get("note"):
            lines.append(f"- Note: {sc['note']}")
        if not cues:
            lines.append("- (the line above carries on over this scene)")
        for c in cues:
            if "variants" in c:
                lines.append(f"- {c['start']:.1f}-{c['end']:.1f} s, one reading per {c['slot']} ending:")
                for e, txt in c["variants"].items():
                    lines.append(f"  - {e}: \"{txt}\" ({words(txt)} words, {words(txt) / (c['end'] - c['start']):.1f} words/s)")
                total_words += max(words(t) for t in c["variants"].values())
            else:
                text = c["text"]
                for k, v in auto.items():
                    text = text.replace("{" + k + "}", str(v))
                n = words(text); total_words += n
                lines.append(f"- {c['start']:.1f}-{c['end']:.1f} s: \"{text}\" ({n} words, {n / (c['end'] - c['start']):.1f} words/s)")
        lines.append("")
    lines += [f"Total: {total_words} words in {spec['total_s']} s ({total_words / spec['total_s']:.2f} words/s, longest reading of each slot counted).", ""]
    return "\n".join(lines)


# ---- main ----------------------------------------------------------------------------------------------------------------------
def render_maps_for(spec: dict, work: Path) -> dict:
    """Map images, lateral views, side by side. The forecast map (source rc) reads results/face_map_RC.json once it exists; the viewers'
    map (source viewers) comes from Sunday's scoring and stays a stamped synthetic placeholder until it is wired in."""
    import render_maps as RM
    rc = next((p for p in RC_FILES if p.exists()), None)
    out, P = {}, RM.parcels()
    for sc in spec["scenes"]:
        for it in sc.get("items", []):
            if it["kind"] == "maps":
                for m in it["maps"]:
                    real = m["source"] == "rc" and rc is not None
                    vals = json.loads(rc.read_text())[m["key"]] if real else RM.synthetic(P, m["seed"])
                    out[m["key"]] = RM.render(RM.to_vertices(vals, P), work / f"map_{m['key']}.png", placeholder=not real,
                                              size=(840, 330), views="lateral")
    return out


def d20_line(spec: dict) -> str:
    """The PREREG D20 pass line, as the notthefall scene carries it."""
    v = next(sc for sc in spec["scenes"] if sc["id"] == "trust_d20")["variants"]["pass"]
    return next(i["text"] for i in v if i["kind"] == "text")


def main(argv=None) -> None:
    import render_video as RV
    ap = argparse.ArgumentParser(); ap.add_argument("--endings", default=""); ap.add_argument("--values", type=Path)
    ap.add_argument("--out", type=Path, default=REPO / "video/v4_placeholder.mp4"); ap.add_argument("--work", type=Path)
    ap.add_argument("--vo-md", type=Path); a = ap.parse_args(argv)
    spec = json.loads(SCENES.read_text())
    if a.vo_md:
        a.vo_md.write_text(vo_markdown(spec)); print(f"{a.vo_md}: written"); return
    endings = dict(x.split("=", 1) for x in a.endings.split(",") if x)
    slots = {sc["slot"]: sc["variants"] for sc in spec["scenes"] if "slot" in sc}
    for s, e in endings.items():
        assert s in slots and e in slots[s], f"no ending {s}={e}"
    values = {**auto_values(), **(json.loads(a.values.read_text()) if a.values else {})}
    work = a.work or Path(tempfile.mkdtemp(prefix="stillhere-v4-")); work.mkdir(parents=True, exist_ok=True)
    endings.setdefault("live", "inset")                                     # the plan; "cut" is the fallback if the inset reads badly
    ctx = {"ghost": ghost_doc(), "film_ghost": film_ghost_doc(), "maps": render_maps_for(spec, work), "d20_line": d20_line(spec),
           "live": {k: cache_entry(spec, k) for k in ("entry", "reversed")},
           "cards": {it["clip"]: story_card_text(it["clip"]) for sc in spec["scenes"] for it in sc.get("items", []) if it["kind"] == "storycard"}}
    ctx["live"]["live"] = ctx["live"]["entry"]
    fps, xf = spec["fps"], spec["crossfade_s"]
    tl = timeline(spec, endings); clips = []
    for i, (sc, start, end) in enumerate(tl):
        secs = end - start + (xf if i < len(tl) - 1 else 0)
        steps = resolve(sc, endings); pngs = []
        for j, (_, items) in enumerate(steps):
            h = work / f"scene_{i:02d}_{j}.html"; h.write_text(scene_html(sc, items, values, spec, ctx))
            p = work / f"scene_{i:02d}_{j}.png"; RV.capture(h, p, work, spec["size"]); pngs.append(p)
        mv = [it for _, items in steps for it in items if it["kind"] in MOVING and (it["kind"] != "take" or values.get("live_take"))]
        moving = moving_clip(mv[0], secs, work, i, values) if mv else None
        c = work / f"scene_{i:02d}.mp4"; scene_clip(pngs, [s for s, _ in steps], secs, moving, c, fps); clips.append(c)
        print(f"scene {i:02d} {sc['id']}: {secs:.1f} s, {len(pngs)} step(s){', moving picture' if mv else ''}", flush=True)
    cues = vo_cues(spec, endings)
    ass, srt = work / "vo.ass", a.out.with_suffix(".srt"); a.out.parent.mkdir(parents=True, exist_ok=True)
    write_subs(cues, values, spec, ass, srt)
    args = ["ffmpeg", "-y", "-loglevel", "error"]
    for c in clips:
        args += ["-i", str(c)]
    fc = [f"[{i}:v]settb=AVTB,fps={fps}[v{i}]" for i in range(len(clips))]
    prev = "v0"
    for i in range(1, len(clips)):
        fc.append(f"[{prev}][v{i}]xfade=transition=fade:duration={xf}:offset={tl[i][1]:.3f}[x{i}]"); prev = f"x{i}"
    if endings["live"] == "inset":
        ins = spec["live"]["inset"]; ip, run_s = inset_clip(spec, values, work)
        args += ["-i", str(ip)]; n = len(clips)
        fc.append(f"[{n}:v]scale={ins['w']}:{ins['h']},fps={fps},setpts=PTS+{ins['start']}/TB[ins]")
        fc.append(f"[{prev}][ins]overlay={ins['x']}:{ins['y']}:eof_action=pass:enable='between(t,{ins['start']},{ins['start'] + run_s:.2f})'[inset]")
        prev = "inset"
    fc.append(f"[{prev}]ass={ass}[out]")
    subprocess.run(args + ["-filter_complex", ";".join(fc), "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "21",
                           "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart", str(a.out)], check=True)
    sheet = a.out.with_name(a.out.stem + "_sheet.png"); mids = [(a0 + b0) / 2 + 1.0 for _, a0, b0 in tl]
    fr = []
    for k, t in enumerate(mids):
        f = work / f"sheet_{k:02d}.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{min(t, spec['total_s'] - 0.2):.2f}", "-i", str(a.out), "-frames:v", "1",
                        "-vf", "scale=480:270", str(f)], check=True); fr.append(f)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(work / "sheet_%02d.png"), "-vf", "tile=6x4:padding=6:color=white",
                    "-frames:v", "1", str(sheet)], check=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(a.out)],
                               capture_output=True, text=True, check=True).stdout)
    print(f"{a.out.name}: {len(clips)} scenes, {dur:.2f} s, {a.out.stat().st_size / 1e6:.1f} MB; endings {endings or 'placeholders'}; "
          f"LPP ghost {'placeholder' if ctx['ghost'].get('placeholder') else 'results/ghostline_lpp_s8.json'}; "
          f"film ghost {'placeholder' if ctx['film_ghost'].get('placeholder') else 'results/ghostline_film.json'}; frames in {work}")


if __name__ == "__main__":
    main()
