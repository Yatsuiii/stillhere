"""Renders the hero-first video (about 3 minutes) from video/scenes_v3.json, with captions word for word from docs/captions.md. Slots
whose result is not known yet (edit check, film maps, story wall, notthefall) hold one panel per possible ending; --endings picks one per
slot and --values fills the {placeholders}. Unfilled placeholders are drawn as yellow tags, so a placeholder cut is never mistaken for a
final one. Film frames come from the edited frame set and always carry the CC BY 3.0 credit. Capture and crossfades reuse
scripts/render_video.py. usage: python scripts/render_video_v3.py [--endings slot=ending,...] [--values VALUES.json] [--frames DIR]
[--out video/hero_placeholder.mp4] [--work DIR]"""
import argparse
import html
import json
import re
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path[:0] = [str(REPO / "scripts"), str(REPO)]
import render_video as RV  # noqa: E402

SCENES = REPO / "video/scenes_v3.json"
DEFAULT_FRAMES = REPO.parent / "stillhere/pilot/probes/k1e_edit/glimmer_edit_frames"
E = html.escape
CSS = """
html,body{margin:0;width:1920px;height:1080px;overflow:hidden;background:#fbfaf7;color:#1d1f23;font-family:Inter,Helvetica,Arial,sans-serif}
main{box-sizing:border-box;width:1920px;height:1080px;padding:64px 104px 48px;display:flex;flex-direction:column;gap:26px;position:relative}
h1{font-size:54px;margin:0;line-height:1.12} .text{font-size:32px;line-height:1.42;max-width:1640px} .small{font-size:22px;color:#5d6470;line-height:1.45;max-width:1680px}
.big{font-size:64px;font-weight:700;line-height:1.2;color:#1f4e79} .beat{font-size:24px;color:#5d6470;border-top:2px solid #e3e0d8;padding-top:10px;max-width:1640px}
.question{font-size:44px;padding:28px 34px;background:#eef2f7;border-radius:14px;border-left:8px solid #1f4e79;max-width:1600px}
table{font-size:28px;border-collapse:collapse} td{padding:8px 28px 8px 0;vertical-align:top} td:first-child{font-weight:600;color:#1f4e79}
.frames{display:flex;gap:24px} .frames figure{margin:0;flex:1} .frames img{width:100%;border-radius:8px} .frames figcaption{font-size:24px;margin-top:6px;text-align:center}
.maps{display:flex;gap:40px} .map{flex:1;height:380px;border:3px dashed #9aa3ad;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:28px;color:#5d6470;text-align:center;padding:20px}
.ph{background:#ffe58a;color:#4a3b00;padding:0 6px;border-radius:5px;font-weight:600} .phbox{border:3px dashed #d4a500;padding:12px 18px;font-size:24px;color:#6b5600;border-radius:10px;max-width:1640px}
.credit{position:absolute;left:104px;bottom:20px;font-size:18px;color:#5d6470} .tag{position:absolute;right:28px;bottom:18px;font-size:16px;color:#9aa3ad}
"""


def fill(text, values):
    out = E(text)
    for k, v in values.items():
        out = out.replace("{" + k + "}", E(str(v)))
    return re.sub(r"\{([A-Za-z_]+)\}", lambda m: f'<span class="ph">{{{m.group(1)}}}</span>', out)


def item(it, values, spec, frames):
    k = it["kind"]
    if k in ("text", "small", "big", "beat", "question"):
        return f'<div class="{k}">{fill(it["text"], values)}</div>'
    if k == "card":
        return "<table>" + "".join(f"<tr><td>{fill(a, values)}</td><td>{fill(b, values)}</td></tr>" for a, b in it["rows"]) + "</table>"
    if k == "frames":
        vs = ("intact", "faces_removed", "area_matched")
        return '<div class="frames">' + "".join(f'<figure><img src="{(frames / v / it["frame"]).as_uri()}"><figcaption>{E(lab)}</figcaption></figure>'
                                               for v, lab in zip(vs, it["labels"])) + "</div>"
    if k == "maps":
        return f'<div class="maps"><div class="map">{fill(it["left"], values)}<br>(map, Sunday)</div><div class="map">{fill(it["right"], values)}<br>(map, Sunday)</div></div>'
    if k == "placeholder":
        return f'<div class="phbox">PLACEHOLDER: {E(it["text"])}</div>'
    if k == "credit":
        return f'<div class="credit">{E(spec["credit"])}</div>'
    raise ValueError(k)


def scene_html(sc, values, endings, spec, frames):
    items = list(sc.get("items", []))
    if "slot" in sc:
        ending = endings.get(sc["slot"])
        items = sc.get("common", []) + (sc["variants"][ending] if ending else
                                        [{"kind": "placeholder", "text": f"{sc['slot']}: one of {', '.join(sc['variants'])} (Sunday)"}])
    has_film = any(i["kind"] in ("frames", "maps") for i in items)
    body = "".join(item(i, values, spec, frames) for i in items)
    if has_film and not any(i["kind"] == "credit" for i in items):
        body += item({"kind": "credit"}, values, spec, frames)
    return f'<!doctype html><meta charset="utf-8"><style>{CSS}</style><main><h1>{fill(sc["title"], values)}</h1>{body}<div class="tag">StillHere</div></main>'


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--endings", default=""); ap.add_argument("--values", type=Path)
    ap.add_argument("--frames", type=Path, default=DEFAULT_FRAMES); ap.add_argument("--out", type=Path, default=REPO / "video/hero_placeholder.mp4")
    ap.add_argument("--work", type=Path); a = ap.parse_args(argv)
    spec = json.loads(SCENES.read_text())
    endings = dict(x.split("=", 1) for x in a.endings.split(",") if x)
    for sc in spec["scenes"]:
        if "slot" in sc and sc["slot"] in endings:
            assert endings[sc["slot"]] in sc["variants"], f"{sc['slot']}: no ending {endings[sc['slot']]}"
    values = json.loads(a.values.read_text()) if a.values else {}
    work = a.work or Path(tempfile.mkdtemp(prefix="stillhere-v3-")); work.mkdir(parents=True, exist_ok=True)
    pngs = []
    for i, sc in enumerate(spec["scenes"]):
        h = work / f"scene_{i:02d}.html"; h.write_text(scene_html(sc, values, endings, spec, a.frames)); p = work / f"scene_{i:02d}.png"
        RV.capture(h, p, work, spec["size"]); pngs.append(p)
    RV.OUT, RV.SHEET = a.out, a.out.with_name(a.out.stem + "_sheet.png")
    a.out.parent.mkdir(exist_ok=True); RV.assemble(pngs, spec)
    print(f"{a.out.name}: {len(pngs)} scenes, {RV.total_seconds(spec):.1f} s; endings {endings or 'placeholders'}; frames in {work}")


if __name__ == "__main__":
    main()
