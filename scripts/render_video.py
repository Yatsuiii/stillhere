"""Renders the silent video skeleton (PLAN §7 storyboard, about 2:45) from video/scenes.json and the same JSON the demo page reads:
results/gate1_real.json (frozen group statistics), results/section8.json, stillhere/validation_table.json, the Pilot results
(conditions check, diarisation, live laptop forecast), demo/lines.json and the Spark desk's REPLAY file. Each scene is one
1920x1080 HTML slide built with the demo page's CSS and SVG builders (scripts/build_demo_page.py, imported, not copied), captured by
headless Chrome, and joined by ffmpeg with 0.5 s crossfades into video/skeleton.mp4 (H.264, 30 fps, no audio track); a contact sheet of
the slides goes to video/contact_sheet.png. Claim text is quoted word for word (PREREG.md with D18, PLAN.md §1, demo/lines.json, the
desk's licensed lines); numbers come from the JSON at render time. Nothing from the privately shared stimulus is read or shown, and the
sedation case study carries only frozen group statistics (no per-volunteer value, subject ID, timecourse or curve).
usage: python scripts/render_video.py [--work DIR]   (frames, Chrome profile and cache go to DIR, a temp dir by default)"""
import argparse
import html
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO)]
import build_demo_page as BP                                         # noqa: E402
from stillhere import desk, egress, frozen                          # noqa: E402

SCENES = REPO / "video/scenes.json"
OUT, SHEET = REPO / "video/skeleton.mp4", REPO / "video/contact_sheet.png"
CHROME = "/usr/bin/google-chrome-stable"
D18_LINE = ("Passive listening tests can show that a brain still processes speech without asking the patient to do anything, which marks "
            "patients worth a full follow-up scan.")
E = html.escape

SLIDE_CSS = BP.CSS + f"""
html,body{{width:1920px;height:1080px;overflow:hidden}}
main.slide{{max-width:none;width:1920px;height:1080px;margin:0;padding:56px 96px 40px;display:flex;flex-direction:column;gap:18px}}
.slide h1{{font-size:50px;line-height:1.15;margin:0}} .slide .tag{{position:absolute;right:28px;bottom:18px;font-size:16px;color:{BP.MUTED}}}
.q{{margin:0;padding:14px 22px;background:{BP.BAND};border-radius:10px;font-size:31px;line-height:1.35}} .q .src{{font-size:19px;margin-top:6px}}
.slide .lines{{margin:0}} .slide .lines li{{font-size:23px;line-height:1.38;margin:8px 0;padding:6px 16px}}
.cap{{font-size:24px}} .small{{font-size:19px;color:{BP.MUTED};line-height:1.4}} .charts{{display:flex;gap:56px;justify-content:center;align-items:flex-start}}
.charts svg{{width:auto}} .ch-flow svg{{width:1600px;height:auto}} .ch-curve svg,.ch-hybrid svg{{height:420px}} .ch-tier svg{{height:340px}}
.slide table{{font-size:24px;max-width:1300px}} .slide .card{{font-size:23px;line-height:1.4}} .slide .card p{{margin:8px 0}}
.refusals{{display:flex;gap:28px}} .refusal{{flex:1;border:2px solid #b42318;border-radius:10px;padding:12px 16px;font-size:21px}}
.refusal b{{color:#b42318;display:block;margin-bottom:4px}} .credits{{font-size:21px;color:{BP.MUTED};max-width:1500px}}
.big{{font-size:40px;line-height:1.3;max-width:1500px}}
"""


def context() -> dict:
    g, s8, lines = BP.load("results/gate1_real.json"), BP.load("results/section8.json"), BP.load("demo/lines.json")
    fz = frozen.Frozen(g)
    card = desk.Desk(fz, replay=REPO / "replay/desk_real.jsonl").ask(desk.public("q_language_fades"))
    return {"g": g, "s8": s8, "lines": lines, "fz": fz, "v": BP.line_values(s8, lines), "card": card,
            "vt": BP.load("stillhere/validation_table.json"), "cm": BP.load("results/conditions_manipulation_check.json"),
            "dz": BP.load("results/diarization_validation.json"), "lt": BP.load("results/live_tribe_librivox.json"),
            "mins": BP.minutes(g["trs_per_state"], g["tr_s"]), "s8m": BP.minutes(s8["trs"], s8["tr_s"])}


def refusals() -> list:
    """Live refusals from the typed egress wrapper (the egress test going red), as (what was tried, the wrapper's own message)."""
    tries = (("A plain string sent as a request", lambda: egress.check("free text typed on the laptop")),
             ("Text that is not on the SHA-256 allowlist", lambda: egress.PublicText(text="free text typed on the laptop")),
             ("A -contributor model id", lambda: egress.Request(model="muse-spark-1.3-contributor", system=[], user=[egress.TierD(values={"n": 1})],
                                                                 schema_name="x", json_schema={"type": "object"})))
    out = []
    for what, fn in tries:
        try:
            fn(); out.append((what, "NOT REFUSED"))
        except ValueError as e:
            msgs = [er["msg"] for er in e.errors()] if hasattr(e, "errors") else [str(e)]
            out.append((what, msgs[0].removeprefix("Value error, ")))
    return out


def sedation_caption(x: dict) -> str:
    g = x["g"]
    return ("Frozen group statistics only (PREREG §6 bars, Gate 1): no per-volunteer value, no subject ID, no timecourse. Included per state: "
            + ", ".join(f'{s.capitalize()} {g["included"][s]}' for s in BP.STATES) + f'. {g["trs_per_state"]} TRs = {x["mins"]} minutes per state.')


def plan_items(x: dict) -> str:
    """The planner slide: declines the validation table forces (not model output) and the conditions with their check."""
    P = [f'<p class="cap"><span class="badge">{E(BP.IN_PROGRESS)}</span>Declines the validation table forces; no planner model output is shown yet.</p><ul class="lines">']
    for ex in x["lines"]["planner_examples"]:
        code, near = BP.expected_decline(x["vt"], ex)
        P.append(f'<li class="fail"><q>{E(ex["request"])}</q> declines with <b>{E(code)}</b>; nearest validated option: {E(near)}.</li>')
    P.append(f'</ul><table><tr><th>Condition ({E(x["cm"]["label"])})</th><th>Words recovered</th><th>Order vs expected</th><th>Result</th></tr>')
    for name, r in x["cm"]["conditions"].items():
        res = ("pass" if r["pass"] else "fail") if r["pass"] is not None else "no word check (reversed speech)"
        P.append(f'<tr><td>{E(name)}</td><td>{r["recall"]:.3f}</td><td>{r["order_vs_expected"]:.3f}</td><td>{E(res)}</td></tr>')
    dz = x["dz"]
    P.append(f'</table><p class="small">Checked with Meta\'s Voice Transcribe; diarisation validated first ({E(dz["label"])}): agreement '
             f'{dz["agreement"]:.3f}, {dz["speaker_changes"]} speaker changes against {dz["true_speaker_changes"]} true.</p>')
    return "".join(P)


def scans_items(x: dict) -> str:
    """The real-scans slide: the pending comparisons and the live laptop forecast."""
    lt = x["lt"]
    P = ['<ul class="lines">'] + [f'<li>{E(t)}: {E(BP.PLACEHOLDERS[k])}</li>' for k, t in BP.SCANS if k in BP.PLACEHOLDERS] + ['</ul>']
    P.append(f'<div class="card"><p class="head">{E(lt["label"])}</p><p>Predicted auditory response strength '
             f'{lt["predicted_aud_response_strength"]:.3f} over {lt["n_aud_parcels"]} auditory parcels; {lt["seconds_predicted"]} s predicted in '
             f'{lt["predict_s"]:.2f} s on a laptop GPU.</p><p>{E(lt["wording"])}</p></div>')
    return "".join(P)


def s8_table(x: dict) -> str:
    s8, st = x["s8"], x["s8"]["stage"]
    rows = (("LANGUAGE, real words", st["language_z"]), ("LANGUAGE, word-shuffled prediction", st["shuffled_z"]),
            ("Best word-level reference (language-model surprisal)", st["surprisal_z"]), ("HEARING", st["hearing_z"]), ("Aligned sound envelope", st["envelope_z"]))
    head = BP.s8(f'Held-out English section 8, {s8["listeners"]} listeners, {s8["trs"]} TRs = {x["s8m"]} minutes', x["lines"])
    return (f'<table><tr><th>{E(head)}</th><th>z</th></tr>'
            + "".join(f"<tr><td>{E(a)}</td><td>{BP.f2(z)}</td></tr>" for a, z in rows) + "</table>")


def quote(text: str, src: str) -> str:
    return f'<blockquote class="q"><q>{E(text)}</q><span class="src">{E(src)}</span></blockquote>'


def item_html(it: dict, x: dict, sid: str) -> str:
    k = it["kind"]
    if k == "prereg":
        return quote(it["text"], "PREREG §7, as reworded by D18" if it["text"] == D18_LINE else "PREREG §7")
    if k == "plan":
        return quote(it["text"] + ("" if it["text"].endswith(".") else "."), f'{it["note"]}; PLAN §1' if it.get("note") else "PLAN §1")
    if k == "lpp_line":
        ln = x["lines"]["lpp"][it["index"]]; q = BP.fill(ln["quote"], x["v"]); note = BP.fill(ln["note"], x["v"]) if ln["note"] else "PREREG §7"
        lab = f' ({x["lines"]["s8_label"]})' if ("section 8" in q or "section 8" in note) and x["lines"]["s8_label"] not in note else ""
        return quote(q, note + lab)
    if k == "m3":
        return f'<p class="small">M3 caveat (PREREG §7): {E(BP.s8(BP.fill(x["lines"]["m3"], x["v"]), x["lines"]))}</p>'
    if k == "desk_line":
        lid = it["line_id"]
        if lid not in desk.licensed_lines(x["fz"]):
            raise ValueError(f"line {lid} is not licensed by the frozen bars")
        cls = "fail" if lid.endswith("fallback") or lid == "sedation_no_line" else "pass"
        return f'<ul class="lines"><li class="{cls}">{E(desk.render(x["fz"], lid))}</li></ul>'
    if k == "timing_note":
        return f'<ul class="lines"><li>{E(desk.TIMING_NOTE)}</li></ul>'
    if k == "chart":
        svg = {"flow": BP.flow_svg, "curve": lambda: BP.curve_svg(x["s8"]), "hybrid": lambda: BP.hybrid_svg(x["s8"]),
               "tier": BP.tier_svg}[it["name"]]()
        return f'<div class="ch-{it["name"]}{" hook" if sid == "hook" else ""}">{svg}</div>'
    if k == "sedation_caption":
        return f'<p class="cap">{E(sedation_caption(x))}</p>'
    if k == "s8_label":
        return f'<p class="small">{E(BP.s8("Held-out LPP English section 8", x["lines"]))}</p>'
    if k == "plan_items":
        return plan_items(x)
    if k == "scans_items":
        return scans_items(x)
    if k == "s8_table":
        return s8_table(x)
    if k == "desk_card":
        head, *body = x["card"].text.split("\n"); q = desk.public("q_language_fades").text.strip()
        return (f'<p class="cap"><span class="badge">{E(x["card"].badge)}</span>Question from the menu: <q>{E(q)}</q></p>'
                f'<div class="card"><p class="head">{E(head)}</p>' + "".join(f"<p>{E(b)}</p>" for b in body) + "</div>")
    if k == "egress_refusals":
        return '<div class="refusals">' + "".join(f'<div class="refusal"><b>Refused: {E(w)}</b>{E(m)}</div>' for w, m in refusals()) + "</div>"
    if k == "caption":
        return f'<p class="{"big" if sid in ("picker", "close") else "cap"}">{E(it["text"])}</p>'
    if k == "credits":
        return ('<p class="credits">TRIBE v2: Meta FAIR (d\'Ascoli et al. 2026), weights and code released under CC BY-NC 4.0. Data: OpenNeuro '
                'ds003643 (Le Petit Prince), ds003171 (propofol sedation, frozen group statistics only), ds002345 (Narratives) and ds004892 '
                '(Emo-FilM); LibriVox (public domain); Tears of Steel (Blender Foundation, CC BY 3.0). Every number comes from a JSON file in '
                'results/ or stillhere/; bars and wording are frozen in PREREG.md. New results are labelled exploratory.</p>')
    raise ValueError(f"unknown scene item kind: {k}")


def scene_html(scene: dict, x: dict) -> str:
    body, row = [], []
    for it in scene["items"]:                                         # consecutive curve and hybrid charts share one row
        h = item_html(it, x, scene["id"])
        if it["kind"] == "chart" and it["name"] in ("curve", "hybrid"):
            row.append(h); continue
        if row:
            body.append(f'<div class="charts">{"".join(row)}</div>'); row = []
        body.append(h)
    if row:
        body.append(f'<div class="charts">{"".join(row)}</div>')
    title = scene["title"]
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{E(title)}</title><style>{SLIDE_CSS}</style></head>'
            f'<body><main class="slide"><h1>{E(title)}</h1>{"".join(body)}<span class="tag">StillHere video skeleton, scene {E(scene["id"])}</span>'
            '</main></body></html>')


def total_seconds(spec: dict) -> float:
    s = spec["scenes"]
    return sum(x["seconds"] for x in s) - spec["crossfade_s"] * (len(s) - 1)


def capture(html_path: Path, png: Path, work: Path, size) -> None:
    env = {**os.environ, "HOME": str(work / "home"), "XDG_CACHE_HOME": str(work / "cache"), "XDG_CONFIG_HOME": str(work / "config")}
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1", "--no-first-run",
                    f"--user-data-dir={work / 'profile'}", f"--disk-cache-dir={work / 'cache'}", f"--window-size={size[0]},{size[1]}",
                    f"--screenshot={png}", html_path.as_uri()], check=True, capture_output=True, env=env, timeout=120)


def assemble(pngs: list, spec: dict) -> None:
    fps, xf, sc = spec["fps"], spec["crossfade_s"], spec["scenes"]
    args = ["ffmpeg", "-y", "-loglevel", "error"]
    for p, s in zip(pngs, sc):
        args += ["-loop", "1", "-framerate", str(fps), "-t", str(s["seconds"]), "-i", str(p)]
    f = [f"[{i}:v]fps={fps},format=yuv420p,settb=AVTB[v{i}]" for i in range(len(pngs))]
    prev, acc = "v0", 0.0
    for i in range(1, len(pngs)):
        acc += sc[i - 1]["seconds"]
        f.append(f"[{prev}][v{i}]xfade=transition=fade:duration={xf}:offset={acc - i * xf:.3f}[x{i}]"); prev = f"x{i}"
    args += ["-filter_complex", ";".join(f), "-map", f"[{prev}]", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart", str(OUT)]
    subprocess.run(args, check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(Path(pngs[0]).parent / "scene_%02d.png"), "-vf",
                    "scale=640:360,tile=4x3:padding=8:color=white", "-frames:v", "1", str(SHEET)], check=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0]); ap.add_argument("--work", type=Path)
    a = ap.parse_args(argv)
    work = a.work or Path(tempfile.mkdtemp(prefix="stillhere-video-")); work.mkdir(parents=True, exist_ok=True)
    spec = json.loads(SCENES.read_text()); x = context(); pngs = []
    for i, sc in enumerate(spec["scenes"]):
        h = work / f"scene_{i:02d}.html"; h.write_text(scene_html(sc, x)); p = work / f"scene_{i:02d}.png"
        capture(h, p, work, spec["size"]); pngs.append(p)
    OUT.parent.mkdir(exist_ok=True); assemble(pngs, spec)
    print(f"{OUT.relative_to(REPO)}: {len(pngs)} scenes, {total_seconds(spec):.1f} s; frames in {work}")


if __name__ == "__main__":
    main()
