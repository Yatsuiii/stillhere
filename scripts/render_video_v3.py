"""Renders the 3-minute video from video/scenes_v3.json (storyline of the ruling of Oct 29, 06:30 UTC), with captions and voice-over word
for word from docs/captions.md. Slots whose result is not known yet (notthefall, the edit check, the film test) hold one panel per
possible ending; --endings picks one per slot (an ending may change the scene's length, or leave the scene out) and --values fills the
{placeholders}. Unfilled placeholders are drawn as yellow tags, so a placeholder cut is never mistaken for a final one. The voice-over
is shown as a subtitle until it is recorded. Every results slide carries "Not a diagnosis or medical advice."; every slide with film
frames or maps carries the CC BY 3.0 credit. Capture and crossfades reuse scripts/render_video.py.
usage: python scripts/render_video_v3.py [--endings slot=ending,...] [--values VALUES.json] [--frames DIR] [--out video/v3_placeholder.mp4]
[--work DIR]"""
import argparse
import html
import json
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO)]
import render_video as RV  # noqa: E402

SCENES = REPO / "video/scenes_v3.json"
DEFAULT_FRAMES = REPO.parent / "stillhere/pilot/probes/k1e_editcheck/edit_check_frames"   # final v2 edit-check frames
E = html.escape
CSS = """
html,body{margin:0;width:1920px;height:1080px;overflow:hidden;background:#fbfaf7;color:#1d1f23;font-family:Inter,Helvetica,Arial,sans-serif}
main{box-sizing:border-box;width:1920px;height:1080px;padding:64px 104px 48px;display:flex;flex-direction:column;gap:26px;position:relative}
h1{font-size:54px;margin:0;line-height:1.12} .text{font-size:32px;line-height:1.42;max-width:1640px} .small{font-size:22px;color:#5d6470;line-height:1.45;max-width:1680px}
.big{font-size:56px;font-weight:700;line-height:1.22;color:#1f4e79;max-width:1660px} .beat{font-size:24px;color:#5d6470;border-top:2px solid #e3e0d8;padding-top:10px;max-width:1640px}
.question{font-size:44px;padding:28px 34px;background:#eef2f7;border-radius:14px;border-left:8px solid #1f4e79;max-width:1600px}
.catch{font-size:30px;padding:18px 24px;border:3px solid #b42318;border-radius:12px;color:#7a1a12;max-width:1600px}
table{font-size:28px;border-collapse:collapse} td{padding:8px 28px 8px 0;vertical-align:top} td:first-child{font-weight:600;color:#1f4e79}
.frames{display:flex;gap:24px} .frames figure{margin:0;flex:1} .frames img{width:100%;border-radius:8px} .frames figcaption{font-size:24px;margin-top:6px;text-align:center}
.maps{display:flex;gap:40px} .map{flex:1;height:340px;border:3px dashed #9aa3ad;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:28px;color:#5d6470;text-align:center;padding:20px}
.ph{background:#ffe58a;color:#4a3b00;padding:0 6px;border-radius:5px;font-weight:600} .phbox{border:3px dashed #d4a500;padding:12px 18px;font-size:24px;color:#6b5600;border-radius:10px;max-width:1640px}
.vo{position:absolute;left:104px;right:104px;bottom:70px;font-size:26px;line-height:1.4;color:#fff;background:rgba(20,24,30,.82);padding:14px 22px;border-radius:10px}
.foot{position:absolute;left:104px;bottom:20px;font-size:18px;color:#5d6470} .tag{position:absolute;right:28px;bottom:18px;font-size:16px;color:#9aa3ad}
"""


def fill(text, values):
    out = E(text)
    for k, v in values.items():
        out = out.replace("{" + k + "}", E(str(v)))
    return re.sub(r"\{([A-Za-z_]+)\}", lambda m: f'<span class="ph">{{{m.group(1)}}}</span>', out)


def item(it, values, frames):
    k = it["kind"]
    if k in ("text", "small", "big", "beat", "question", "catch"):
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
    raise ValueError(k)


def resolve(sc, endings):
    """(items, seconds) for a scene under the chosen endings; None when the chosen ending leaves the scene out."""
    if "slot" not in sc:
        return sc.get("items", []), sc["seconds"]
    ending = endings.get(sc["slot"])
    if ending is None:
        return sc.get("common", []) + [{"kind": "placeholder", "text": f"{sc['slot']}: one of {', '.join(sc['variants'])} (Sunday)"}], sc["seconds"]
    v = sc["variants"][ending]
    if v is None:
        return None
    items, secs = (v["items"], v.get("seconds", sc["seconds"])) if isinstance(v, dict) else (v, sc["seconds"])
    return sc.get("common", []) + items, secs


def scene_html(sc, items, values, spec, frames):
    body = "".join(item(i, values, frames) for i in items)
    foot = []
    if sc.get("film") or any(i["kind"] in ("frames", "maps") for i in items):    # film results slides always carry the credit
        foot.append(spec["credit"])
    if sc.get("results"):
        foot.append(spec["nd"])
    vo = f'<div class="vo">{E(sc["vo"])}</div>' if sc.get("vo") else ""
    f = f'<div class="foot">{E("  ".join(foot))}</div>' if foot else ""
    return f'<!doctype html><meta charset="utf-8"><style>{CSS}</style><main><h1>{fill(sc["title"], values)}</h1>{body}{vo}{f}<div class="tag">StillHere</div></main>'


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--endings", default=""); ap.add_argument("--values", type=Path)
    ap.add_argument("--frames", type=Path, default=DEFAULT_FRAMES); ap.add_argument("--out", type=Path, default=REPO / "video/v3_placeholder.mp4")
    ap.add_argument("--work", type=Path); a = ap.parse_args(argv)
    spec = json.loads(SCENES.read_text())
    endings = dict(x.split("=", 1) for x in a.endings.split(",") if x)
    for sc in spec["scenes"]:
        if "slot" in sc and sc["slot"] in endings:
            assert endings[sc["slot"]] in sc["variants"], f"{sc['slot']}: no ending {endings[sc['slot']]}"
    values = json.loads(a.values.read_text()) if a.values else {}
    work = a.work or Path(tempfile.mkdtemp(prefix="stillhere-v3-")); work.mkdir(parents=True, exist_ok=True)
    pngs, shown = [], []
    for sc in spec["scenes"]:
        r = resolve(sc, endings)
        if r is None:
            continue
        items, secs = r; i = len(shown)
        h = work / f"scene_{i:02d}.html"; h.write_text(scene_html(sc, items, values, spec, a.frames)); p = work / f"scene_{i:02d}.png"
        RV.capture(h, p, work, spec["size"]); pngs.append(p); shown.append({**sc, "seconds": secs})
    run = {**spec, "scenes": shown}
    RV.OUT, RV.SHEET = a.out, a.out.with_name(a.out.stem + "_sheet.png")
    a.out.parent.mkdir(exist_ok=True); RV.assemble(pngs, run)
    print(f"{a.out.name}: {len(pngs)} scenes, {RV.total_seconds(run):.1f} s; endings {endings or 'placeholders'}; frames in {work}")


if __name__ == "__main__":
    main()
