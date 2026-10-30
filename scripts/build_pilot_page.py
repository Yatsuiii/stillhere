"""Builds demo/index.html, the StillHere Pilot page judges open: one static, replay-only page (no model call, no key, no network, no
script) that walks one demo study through StillHere, with wording from docs/captions.md and every number from a committed JSON file.

  1. Pick a demo study (eval/demo_requests.json; the two typed asks of docs/captions.md until it lands).
  2. The recorded plan: Muse Spark 1.3's card per study (results/demo_runs.json, recorded live, replayed here; a labelled pending
     slot until it lands), and the planner's held-out false-accept rates (results/planner_v2_eval.json).
  3. The LibriVox conditions with players and their Voice Transcribe check (results/conditions_manipulation_check.json); the word
     scramble is caught as broken before scanning.
  4. The SAM film edits (intact, faces filled, same-size control) at two instants, with the Spark Open edit check
     (results/edit_check_spark_open.json) and Glimmer's slot (results/edit_check_glimmer.json, Saturday).
  5. The laptop TRIBE forecast for the story study's two versions (results/demo_live_tribe_conditions.json, with the first live run
     in results/live_tribe_librivox.json): a forecast, not checked against scans.
  6. Did the forecasts hold up? LPP section 8 (results/section8.json) with its "what it had to beat" row (results/
     section8_baselines.json); slots for the story wall (results/story_wall.json), notthefall (results/d20_notthefall.json) and the film
     (results/face_map.json, results/emofilm_film_score.json). A missing file is a labelled pending slot.
  The sedation case study is one link to docs/writeup_sedation.md. Stored card wording is shown in plain words (DISPLAY). Media come from demo/media/ (scripts/build_pilot_media.py, built once).
Standard library only, so a clean clone builds it. The build refuses banned wording.
usage: python scripts/build_pilot_page.py [--single OUT.html]   (--single also writes a copy with the media inlined, for hosting)"""
import argparse
import base64
import datetime
import html
import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "demo/index.html"
MEDIA = REPO / "demo/media"
E = html.escape
ND = "Not a diagnosis or medical advice."
CREDIT = "Tears of Steel (CC) Blender Foundation | mango.blender.org, CC BY 3.0. Edited for this study."
S8_LABEL = "recomputed with code built during the event; first measured Oct 22-23"
REPO_URL = "https://github.com/Yatsuiii/stillhere"                 # the only remote link: an anchor to the repository, never a resource
SEDATION_URL = f"{REPO_URL}/blob/main/docs/writeup_sedation.md"
BANNED = ("aware", "awareness", "conscious", "consciousness", "hospital", "detect", "detects", "detected", "detection", "detections",
          "diagnose", "a test for patients", "before anyone looked", "face check", "detects faces", "blur", "validated", "understanding",
          "comprehension")
SLOTS = {"demo_requests": "eval/demo_requests.json", "demo_runs": "results/demo_runs.json", "glimmer": "results/edit_check_glimmer.json",
         "story_wall": "results/story_wall.json", "d20": "results/d20_notthefall.json", "face_map": "results/face_map.json",
         "film_score": "results/emofilm_film_score.json", "live_conditions": "results/demo_live_tribe_conditions.json"}
ASKS = ({"id": "story", "text": "Build a test from this new story, with a scrambled control."},          # docs/captions.md c.1 and d.1
        {"id": "film", "text": "If the faces in this film scene are filled in, what changes in the brain?"})
BACKENDS = {"spark-1.3": "Muse Spark 1.3 (Meta Model API), the default planner", "spark-open": "Muse Spark Open (Fireworks)",
            "spark-open-baseten": "Muse Spark Open (Baseten)"}
COND_NAMES = {"intact": "Intact story", "sentence_scramble": "Sentence scramble", "word_scramble": "Word scramble", "reversed": "Reversed"}
VERSION_NAMES = {"intact": "Intact", "faces_removed": "Faces filled", "area_matched": "Same-size control"}
INK, MUTED, ACCENT, PASS, FAIL, WAIT = "#1b1f24", "#4a5560", "#1f4e79", "#1a6b34", "#a8261b", "#7a5200"
CSS = f"""
*{{box-sizing:border-box}} html{{-webkit-text-size-adjust:100%}}
body{{margin:0;background:#fbfaf7;color:{INK};font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
main{{max-width:880px;margin:0 auto;padding:20px 16px 40px}}
h1{{font-size:1.9rem;margin:.2em 0 .1em;line-height:1.15}} h2{{font-size:1.3rem;margin:2.2em 0 .5em;padding-top:.6em;border-top:2px solid #e3e0d8}}
h3{{font-size:1.05rem;margin:1.3em 0 .4em}} p{{margin:.5em 0}} q{{quotes:"\\201C" "\\201D"}}
.lead{{font-size:1.06rem}} .muted,.small{{color:{MUTED}}} .small{{font-size:.88rem}}
nav{{display:flex;flex-wrap:wrap;gap:6px;margin:14px 0}} nav a{{font-size:.85rem;color:{ACCENT};text-decoration:none;border:1px solid #c9d3de;
border-radius:999px;padding:3px 10px;background:#fff}}
.panel{{background:#fff;border:1px solid #e3e0d8;border-radius:12px;padding:14px 16px;margin:12px 0}}
.nd{{font-size:.85rem;color:{MUTED};border-top:1px solid #eee;margin-top:10px;padding-top:6px}}
.credit{{font-size:.8rem;color:{MUTED}}}
.badge{{display:inline-block;font-size:.78rem;font-weight:600;border-radius:999px;padding:1px 9px;margin-right:6px;vertical-align:1px}}
.b-pass{{background:#e6f4ea;color:{PASS}}} .b-fail{{background:#fdecea;color:{FAIL}}} .b-wait{{background:#fff4d6;color:{WAIT}}}
.b-info{{background:#e8eef5;color:{ACCENT}}}
.pending{{border:1px dashed #d9c48a;background:#fffbf0}}
.big{{font-size:1.25rem;font-weight:650;color:{ACCENT};line-height:1.35}}
.beat{{font-size:.9rem;color:{MUTED};border-left:3px solid #c9d3de;padding-left:10px}}
.picker{{display:flex;flex-direction:column;gap:8px}} .picker input{{position:absolute;opacity:0;pointer-events:none}}
.picker label{{display:block;cursor:pointer;background:#fff;border:1px solid #c9d3de;border-radius:10px;padding:9px 12px}}
.picker label:hover{{border-color:{ACCENT}}} .plan{{display:none}}
.card{{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#f4f6f8;
border-radius:8px;padding:10px 12px;margin:6px 0}}
.conds{{display:grid;grid-template-columns:1fr;gap:10px}} @media(min-width:700px){{.conds{{grid-template-columns:1fr 1fr}}}}
.cond h3{{margin:.1em 0 .3em}} audio{{width:100%;height:40px}}
.caught{{border:2px solid {FAIL}}}
.frames{{display:grid;grid-template-columns:1fr;gap:8px}} @media(min-width:640px){{.frames{{grid-template-columns:repeat(3,1fr)}}}}
figure{{margin:0}} img{{width:100%;height:auto;display:block;border-radius:6px}} figcaption{{font-size:.85rem;color:{MUTED};margin-top:3px}}
.tbl{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:.9rem}} th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid #eee;
vertical-align:top}} th{{color:{MUTED};font-weight:600}}
details{{margin:12px 0}} summary{{cursor:pointer;font-weight:600;color:{ACCENT}}}
footer{{margin-top:2.5em;border-top:2px solid #e3e0d8;padding-top:10px;font-size:.85rem;color:{MUTED}}} footer li{{margin:.3em 0}}
svg{{max-width:100%;height:auto}} code{{overflow-wrap:anywhere;font-size:.92em}}
"""


def load(rel: str):
    p = REPO / rel
    return json.loads(p.read_text()) if p.is_file() else None


def pct(x: float) -> str:
    return f"{round(100 * x)}%"


def pnum(p: float) -> str:
    return f"{p:.2g}"


def badge(kind: str, text: str) -> str:
    return f'<span class="badge b-{kind}">{E(text)}</span>'


def panel(body: str, cls: str = "", nd: bool = True, credit: bool = False, id: str = "") -> str:
    tail = (f'<p class="credit">{E(CREDIT)}</p>' if credit else "") + (f'<p class="nd">{E(ND)}</p>' if nd else "")
    attr = f' id="{id}"' if id else ""
    return f'<div class="panel {cls}"{attr}>{body}{tail}</div>'


def pending(title: str, when: str, rel: str, note: str = "") -> str:
    body = (f'<h3>{badge("wait", "pending")}{E(title)}</h3><p>{E(when)}</p>{f"<p class=small>{E(note)}</p>" if note else ""}'
            f'<p class="small">Fills from <code>{E(rel)}</code> when {"they land" if " and " in rel else "it lands"}.</p>')
    return panel(body, "pending")


# ---- 1 and 2: pick a study, the recorded plan ------------------------------------------------------------------------------------
REASONS = {"analysis_not_offered": "analysis not offered", "clinical_out_of_scope": "a clinical question, out of scope"} | \
    {f"{a}_not_validated": f"{a.replace('_', ' ')} not on the tested menu" for a in ("population", "language", "brain_area", "condition", "stimulus")}
DISPLAY = ((re.compile(r"Nearest validated option: "), "Nearest tested option: "),             # planner v2 is frozen; its stored text keeps
           (re.compile(r"from the validation table"), "from its table of tested options"),     # these words, the page shows plain ones
           (re.compile(r"\bnot validated\b"), "not checked against scans"),
           (re.compile(r"(nothing about language) or understanding\b"), r"\1"))
DISPLAY_NOTE = ("Display wording: the stored cards use code names (for example condition_not_validated) and older labels; this page shows "
                "them in plain words. The records are unchanged in results/demo_runs.json.")


def plain(text: str) -> str:
    """Stored wording in plain words: decline codes, the planner's fixed sentences, the forecast label."""
    text = re.sub(r"Declined: ([a-z_]+)\.", lambda m: f"Declined: {REASONS.get(m.group(1), m.group(1).replace('_', ' '))}.", text)
    for rx, new in DISPLAY:
        text = rx.sub(new, text)
    return text


def studies(reqs, runs) -> list:
    """Demo requests (eval/demo_requests.json: a list, or {"requests"|"questions": [...]}; until it lands, the two typed asks). A request
    with "reworded_from" is shown under the request it rewords, not as a study of its own."""
    items = (reqs.get("requests") or reqs.get("questions")) if isinstance(reqs, dict) else reqs
    if not items and runs:
        seen = {}
        for r in runs:
            seen.setdefault(run_id(r), {"id": run_id(r), "text": r.get("text") or r.get("request") or r.get("question") or run_id(r)})
        items = list(seen.values())
    norm = []
    for i, it in enumerate(items or ASKS):
        it = {"text": it} if isinstance(it, str) else it
        norm.append({**it, "id": str(it.get("id") or it.get("qid") or i), "text": it.get("text") or it.get("request") or it.get("question") or ""})
    out = [dict(s, rewordings=[]) for s in norm if not s.get("reworded_from")]
    by_id = {s["id"]: s for s in out}
    for s in norm:
        if s.get("reworded_from") in by_id:
            by_id[s["reworded_from"]]["rewordings"].append(s)
    return out


def run_list(d) -> list:
    return (d.get("runs") or []) if isinstance(d, dict) else (d or [])


def run_id(r: dict) -> str:
    return str(r.get("qid") or r.get("id") or r.get("request_id") or "")


def card_lines(r: dict) -> list:
    """The recorded card, line by line. Display rules: a manipulation-check line's JSON is summarised per condition (transcripts left
    out; the full card stays in the results file); a structured "nearest" field, when a record has one, replaces the stored nearest-option
    sentence; and every line is shown in plain words (plain(); planner v2 is frozen, so its stored text keeps its own words)."""
    v = next((r[k] for k in ("card", "card_text", "display") if isinstance(r.get(k), (str, dict))), "")
    v = (v.get("text") or "") if isinstance(v, dict) else v
    if not v.strip():
        spec = r.get("spec") or {}
        v = (f"Declined: {r.get('reason') or spec.get('decline_reason')}." if r.get("decision") == "decline" else
             "StudySpec: " + ", ".join(f"{k}={spec[k]}" for k in ("population", "language", "stimulus", "analysis") if k in spec) + ".")
    out = []
    for ln in v.strip().split("\n"):
        head, sep, body = ln.partition(": {")
        if sep and head.startswith("Manipulation check"):
            try:
                checks = json.loads("{" + body.rstrip(".").rstrip())
                ln = f"{head}: " + "; ".join(f"{c} kept {pct(x['recall'])} of its words, "
                                             + {True: "pass", False: "fail"}.get(x.get("pass"), "no word check") for c, x in checks.items()) + "."
            except (ValueError, KeyError, TypeError, AttributeError):
                pass
        if r.get("decision") == "decline" and r.get("nearest") and ln.startswith("Declined:"):
            sub = f"Nearest validated option: {r['nearest']}."
            ln = re.sub(r"Nearest validated option: [^.]*\.", sub, ln) if "Nearest validated option:" in ln else f"{ln} {sub}"
        out.append(plain(ln))
    return out


def decision_badge(r: dict) -> str:
    dec = r.get("decision")
    return badge("pass", "plans it") if dec == "accept" else badge("info", "declines, with its reason") if dec == "decline" else \
        badge("wait", str(dec or "no answer"))


def plan_card(r: dict, head: bool = True) -> str:
    tag = f'<p>{decision_badge(r)}{E(BACKENDS.get(r.get("backend"), str(r.get("backend") or "")))}</p>' if head else ""
    secs = r.get("total_s") or sum(c.get("s") or 0 for c in r.get("calls") or [])
    meta = " | ".join(m for m in (f"{secs:.1f} s live" if secs else "", "tools run: " + ", ".join(r["tools"]) if r.get("tools") else "") if m)
    stim = (r.get("spec") or {}).get("stimulus")
    note = ("The demo's tools hold only the LibriVox clip, so the tool lines above describe that clip, not " + stim + "."
            if stim and stim != "librivox_audiobook" and set(r.get("tools") or []) & {"make_conditions", "check_conditions", "forecast"} else "")
    return (f'{tag}<div class="card">{E(chr(10).join(card_lines(r)))}</div>' + (f'<p class="small">{E(note)}</p>' if note else "")
            + (f'<p class="small">{E(meta)}</p>' if meta else ""))


def cards_for(runs: list, st: dict) -> dict:
    """The first recorded run per backend for one request, Spark 1.3 first."""
    mine = sorted((r for r in runs if run_id(r) == st["id"]), key=lambda r: (r.get("backend") != "spark-1.3", r.get("repeat") or 0))
    first = {}
    for r in mine:
        first.setdefault(r.get("backend"), r)
    return first


def cards_html(first: dict) -> str:
    return plan_card(first["spark-1.3"]) + "".join(f'<details><summary>{E(BACKENDS.get(b, b))}: {decision_badge(r)}</summary>'
                                                  f'{plan_card(r, head=False)}</details>' for b, r in first.items() if b != "spark-1.3")


def planner_rates() -> dict:
    """Held-out false accepts per backend (results/planner_v2_eval.json), with the exact (Clopper-Pearson) 95% upper bound."""
    d, qs = load("results/planner_v2_eval.json"), load("eval/planner_questions_v2_heldout.json")
    Q = {q["id"]: q for q in qs["questions"]}; out = {}
    for b in d["models"]:
        dec = [r for r in d["runs"] if r["backend"] == b and r["set"] == "heldout" and r["qid"] in Q
               and Q[r["qid"]]["expected"]["decision"] == "decline"]
        k = sum(r["decision"] == "accept" for r in dec); out[b] = (k, len(dec), upper95(k, len(dec)))
    return out


def upper95(k: int, n: int) -> float:
    """Exact two-sided 95% upper bound for k of n: the p where P(X <= k) = 0.025 (bisection; the same value as scipy's beta.ppf)."""
    if k == n:
        return 1.0
    cdf = lambda p: sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
    lo, hi = k / n, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if cdf(mid) > 0.025 else (lo, mid)
    return (lo + hi) / 2


def planner_line(rates: dict) -> str:
    k, n, up = rates["spark-1.3"]; fw, bt = rates["spark-open"], rates["spark-open-baseten"]
    f = lambda t: f"{100 * t[0] / t[1]:.1f}%"
    return ("Ask for something it cannot support and it says so, with the reason and the nearest thing it can do. On requests it had "
            f"never seen, the default planner (Muse Spark 1.3) accepted {f(rates['spark-1.3'])} it should have declined (upper bound "
            f"{100 * up:.1f}%). Private mode, Muse Spark Open, which can run on your own servers: {f(fw)} (Fireworks), {f(bt)} (Baseten).")


def picker_html(sts: list, runs_doc) -> tuple:
    runs = run_list(runs_doc); P, css = ['<div class="picker">'], []
    for i, st in enumerate(sts):
        shows = f'<br><span class="small">Shows: {E(st["shows"])}</span>' if st.get("shows") else ""
        P.append(f'<input type="radio" name="study" id="st{i}"{" checked" if i == 0 else ""}><label for="st{i}"><q>{E(st["text"])}</q>{shows}</label>')
        css.append(f"#st{i}:checked~.plans #plan{i}{{display:block}}#st{i}:checked+label{{border:2px solid {ACCENT};background:#eef2f7}}")
    P.append('<div class="plans" id="plan">')
    for i, st in enumerate(sts):
        first = cards_for(runs, st)
        P.append(f'<div class="plan" id="plan{i}"><h2>2. The recorded plan</h2>')
        if "spark-1.3" in first:
            body = [f'<p class="beat">Replayed from a live run, once per backend; this page calls no model. <code>{SLOTS["demo_runs"]}</code>'
                    f'{" (" + E(runs_doc["label"]) + ")" if isinstance(runs_doc, dict) and runs_doc.get("label") else ""}.</p>', cards_html(first)]
            for rw in st["rewordings"]:
                rf = cards_for(runs, rw)
                if "spark-1.3" in rf:
                    body.append(f'<h3>Round {E(str(rw.get("round") or 2))}, reworded: <q>{E(rw["text"])}</q></h3>'
                                + (f'<p class="small">{E(rw["why"])}.</p>' if rw.get("why") else "") + cards_html(rf))
            body.append(f'<p class="small">{E(DISPLAY_NOTE)}</p>')
            P.append(panel("".join(body), nd=False))
        else:
            P.append(pending("Muse Spark 1.3's recorded plan for this request", "Recorded live and replayed here once the run lands.",
                             SLOTS["demo_runs"]))
        P.append("</div>")
    P.append("</div></div>")
    return "".join(P), "".join(css)


# ---- 3: the conditions and their check ------------------------------------------------------------------------------------------
def conditions_html(cm: dict) -> str:
    c = cm["conditions"]; i, s, w = c["intact"], c["sentence_scramble"], c["word_scramble"]
    assert i["pass"] and s["pass"] and w["pass"] is False, "the conditions check no longer matches its captions"
    P = [f'<p class="big">Muse Voice Transcribe checks every version before anyone is scanned: the intact story kept {pct(i["recall"])} of '
         f'its words; the sentence scramble kept {pct(s["recall"])} of its words in a new order.</p><div class="conds">']
    for name, r in c.items():
        if name == "intact":
            res = f'{badge("pass", "pass")}kept {pct(r["recall"])} of its words, in order'
        elif name == "sentence_scramble":
            res = f'{badge("pass", "pass")}kept {pct(r["recall"])} of its words; order against the original {r["order_vs_original"]:.2f}'
        elif name == "word_scramble":
            res = f'{badge("fail", "caught as broken")}kept only {pct(r["recall"])} of its words'
        else:
            res = f'{badge("info", "control")}reversed speech: no word check ({r["n_words_heard"]} words heard)'
        P.append(f'<div class="panel cond{" caught" if r["pass"] is False else ""}"><h3>{E(COND_NAMES.get(name, name))}</h3>'
                 f'<audio controls preload="metadata" src="media/alice_75s_{name}.mp3"></audio><p class="small">{res}</p></div>')
    P.append(f'</div><p class="big">The word scramble kept only {pct(w["recall"])} of its words, so it was caught as broken before scanning '
             'and is not used.</p>')
    P.append(f'<p class="small">LibriVox, <i>Alice\'s Adventures in Wonderland</i>, chapter 1, read by Kristen McQuillin (public domain), '
             f'75 s. Seed {cm["seed"]}; rules and transcripts in <code>results/conditions_manipulation_check.json</code> '
             f'({E(cm["label"])}).</p>')
    return panel("".join(P))


# ---- 4: the film edits and their check ------------------------------------------------------------------------------------------
def ks(ec: dict) -> dict:
    """Yes counts per version over both sets (by_set holds [yes, no, n]), and N."""
    bs = ec["rates"]["by_set"]
    return {v: sum(bs[s][v][0] for s in bs) for v in ("intact", "faces_removed", "area_matched")} | \
        {"N": sum(bs[s]["intact"][2] for s in bs)}


def edit_check_line(who: str, ec: dict) -> str:
    k = ks(ec)
    return (f"Edit check: {who} saw a face in {k['intact']} of {k['N']} intact frames, in {k['faces_removed']} of {k['N']} filled ones, "
            f"and in {k['area_matched']} of {k['N']} control frames.")


def failed_bars(ec: dict) -> str:
    r, p = ec.get("rates") or {}, ec.get("pass") or (ec.get("rates") or {}).get("pass") or {}
    names = {"intact": ("intact_with_face_yes", "intact frames called a face"), "faces_removed": ("faces_removed_no", "filled frames called no face"),
             "area_matched": ("area_matched_yes", "control frames called a face"), "no_mask": ("no_mask_yes_or_unparsed", "frames without a face called a face")}
    bad = [f"{names[k][1]}: {r[names[k][0]][0]} of {r[names[k][0]][1]}" for k in names if p.get(k) is False and names[k][0] in r]
    return "; ".join(bad) or str(ec.get("why") or "Glimmer could not run in its slot")


def film_html(media: dict, spark: dict, glimmer) -> str:
    rows = {}
    for it in media["items"]:
        if "frame" in it:
            rows.setdefault(it["frame"], {})[it["version"]] = it
    ans = {(r["name"], r["version"]): r["answer"] for r in spark["rows"]}
    P = ['<p class="big">SAM filled the faces it masked with the surrounding colour. The control fills the same area somewhere else, with '
         'exactly the same fill.</p>']
    for frame, vs in rows.items():
        P.append('<div class="frames">')
        for v in ("intact", "faces_removed", "area_matched"):
            a = ans.get((frame, v))
            P.append(f'<figure><img src="media/{E(vs[v]["file"])}" alt="{E(VERSION_NAMES[v])} frame" loading="lazy" width="640" height="267">'
                     f'<figcaption>{E(VERSION_NAMES[v])}{f"; Spark Open: face? {E(a)}" if a else ""}</figcaption></figure>')
        P.append(f'</div><p class="small">Film frame {E(frame)}.</p>')
    r = spark["rates"]; ok = r["pass"]["all"]
    P.append(f'<h3>{badge("pass" if ok else "fail", "passes all four bars" if ok else "misses a bar")}{E(edit_check_line("Spark Open", spark))}</h3>'
             f'<p class="small">Muse Spark Open on Baseten, asked "Is any human face recognisable? Answer yes or no." about each frame (bars: '
             f'at least 80% yes on intact, 90% no on filled, 80% yes on controls, at most 3 of 31 frames without a face called a face: '
             f'{r["no_mask_yes_or_unparsed"][0]} of {r["no_mask_yes_or_unparsed"][1]}). Reported, not deciding; '
             f'<code>results/edit_check_spark_open.json</code> ({E(spark["label"])}).</p>')
    body = "".join(P)
    if glimmer is None:
        gl = pending("Glimmer's edit check, which decides", "Saturday, same frames and bars. If Glimmer fails any bar, or cannot run in its "
                     "slot, the film face test is NOT RUN.", SLOTS["glimmer"])
    elif (glimmer.get("pass") or glimmer.get("rates", {}).get("pass") or {}).get("all"):
        gl = panel(f'<h3>{badge("pass", "decides: pass")}{E(edit_check_line("Spark Open and Glimmer", glimmer))}</h3>'
                   f'<p class="small">Glimmer\'s counts; <code>{SLOTS["glimmer"]}</code>.</p>')
    else:
        gl = panel(f'<h3>{badge("fail", "decides: fail")}{E(f"The edit check failed ({failed_bars(glimmer)}), so the film test did not run.")}</h3>')
    return panel(body, credit=True) + gl


# ---- 5: the forecast --------------------------------------------------------------------------------------------------------------
COND_COLOURS = {"intact": ACCENT, "sentence_scramble": "#b4501a"}                   # both >= 4.5:1 on white


def timecourse_svg(lc: dict) -> str:
    """Per second, the mean over the auditory parcels of TRIBE's audio-only prediction, one line per condition; each 18.75 s window
    was predicted on its own, so the line breaks between windows."""
    W, H, L, R, T, B = 560, 260, 30, 14, 34, 34
    series = {}
    for r in lc["runs"]:
        pts = [(r["window_s"][0] + j, sum(row) / len(row)) for j, row in enumerate(r["aud_parcel_timecourse"])]
        series.setdefault(r["condition"], []).append(pts)
    ys = [y for segs in series.values() for seg in segs for _, y in seg]; lo, hi = min(ys), max(ys); pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad; tmax = max(r["window_s"][1] for r in lc["runs"])
    X = lambda x: L + (W - L - R) * x / tmax
    Y = lambda y: T + (H - T - B) * (hi - y) / (hi - lo)
    P = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="TRIBE predicted hearing response over time, intact '
         f'and sentence-scrambled">', f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="#9aa3ad"/>']
    for tick in range(0, int(tmax) + 1, 15):
        P.append(f'<text x="{X(tick):.1f}" y="{H - B + 24}" font-size="17" fill="{MUTED}" text-anchor="{'end' if tick + 15 > tmax else 'middle'}">{tick} s</text>')
    for v in (0.0,) if lo < 0 < hi else ():
        P.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W - R}" y2="{Y(v):.1f}" stroke="#d5d9de" stroke-dasharray="3 3"/>'
                 f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" font-size="17" fill="{MUTED}" text-anchor="end">0</text>')
    for k in range(1, len(series[next(iter(series))])):
        x = X(k * tmax / len(series[next(iter(series))]))
        P.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H - B}" stroke="#eceef1"/>')
    for c, segs in series.items():
        for seg in segs:
            P.append(f'<polyline fill="none" stroke="{COND_COLOURS.get(c, INK)}" stroke-width="2.5" points="'
                     + " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in seg) + '"/>')
    for i, c in enumerate(series):
        P.append(f'<rect x="{L + i * 200}" y="{T - 22}" width="18" height="5" fill="{COND_COLOURS.get(c, INK)}"/>'
                 f'<text x="{L + 24 + i * 200}" y="{T - 15}" font-size="17" fill="{INK}">{E(COND_NAMES.get(c, c))}</text>')
    return "".join(P) + "</svg>"


def forecast_html(lt: dict, lc) -> str:
    first = (f'<p class="small">First live run, Oct 28: a {lt["clip_seconds"]:.0f} s clip of the same chapter, predicted in '
             f'{lt["predict_s"]:.2f} s, peak {lt["peak_vram_gb"]:.2f} GB of GPU memory; response strength '
             f'{lt["predicted_aud_response_strength"]:.3f}. <code>results/live_tribe_librivox.json</code> ({E(plain(lt["label"]))}).</p>')
    if lc is None:
        body = (f'<p>{badge("wait", "a forecast, not checked against scans")}Live on a laptop ({E(lt["device"])}): {E(lt["model"])}.</p>'
                f'<p>{E(plain(lt["wording"]))}</p>' + first)
    else:
        rs = lc["runs"]; m = lc["mean_aud_response_strength"]
        pred = sorted(r["predict_s"] for r in rs)
        body = (f'<p>{badge("wait", "a forecast, not checked against scans")}Live on this laptop ({E(rs[0]["device"])}), {E(rs[0]["model"])}, for the '
                f'story study\'s two versions: each 75 s clip in four {rs[0]["clip_seconds"]} s windows, {len(rs)} predictions, '
                f'{pred[0]:.1f}-{pred[-1]:.1f} s each once loaded; peak {max(r["peak_vram_gb"] for r in rs):.2f} GB of GPU memory.</p>'
                f'<div class="tbl">{timecourse_svg(lc)}</div><p class="small">Per second, the mean over {rs[0]["n_aud_parcels"]} auditory '
                f'parcels of the predicted response (raw TRIBE units); each window was predicted on its own, so the lines break between '
                'windows.</p><p>Predicted response strength (hearing, audio-only): '
                + ", ".join(f"{E(COND_NAMES.get(c, c).lower())} <b>{v:.3f}</b>" for c, v in m.items())
                + f' (raw TRIBE units). {E(plain(lc["wording"]))}</p><p class="small"><code>results/demo_live_tribe_conditions.json</code> '
                f'({E(plain(lc["label"]))}).</p>' + first)
    return panel('<p class="big">TRIBE v2 forecasts the healthy-adult response.</p>' + body)


# ---- 6: did the forecasts hold up? ------------------------------------------------------------------------------------------------
def dots_svg(point: float, claim: int, n: int = 15) -> str:
    """One dot per volunteer scan, 1..n: solid up to the claim, lighter up to the point estimate."""
    c = []
    for i in range(1, n + 1):
        fill = ACCENT if i <= claim else "#8fb0d0" if i <= round(point) else "none"
        c.append(f'<circle cx="{i * 22 - 8}" cy="14" r="8" fill="{fill}" stroke="{ACCENT}" stroke-width="1.5"/>')
    return (f'<svg viewBox="0 0 {n * 22 + 4} 28" width="{n * 22 + 4}" height="28" role="img" aria-label="about {round(point)} of {n} '
            f'volunteer scans, at least {claim}">{"".join(c)}</svg>')


def lpp_html(s8: dict, b: dict, ghost: dict = None) -> str:
    ve = s8["volunteer_equivalent_language"]; assert s8["label"] == S8_LABEL == b["section8_label"]
    beats = [b[w][n]["tribe_vs"][x]["verdict"] for w in ("145", "full") for n in ("LANG", "AUD") for x in ("word_rate", "envelope")]
    k = sum(v == "TRIBE beats" for v in beats); wr = b["145"]["LANG"]["tribe_vs"]["word_rate"]
    two = (f'Two numbers for the same {s8["listeners"]} listeners: the held-out language result is {s8["stage"]["language_z"]:.2f} z, '
           'with word rate and sound partialled out of the forecast as the language gauge does (results/section8.json); the comparison with '
           f'word rate and the sound envelope scores the forecast unpartialled, {b["145"]["LANG"]["mean_z"]["tribe"]:.2f} z over the same '
           f'{s8["trs"]} time points, so that every predictor is scored the same way (results/section8_baselines.json).')
    g = ""
    if ghost:
        g = (f'<div class="tbl">{ghost_svg(ghost, ("group", "one"))}</div><p class="small">{E(ghost["chip"]["long"])}. '
             f'{E(ghost["frozen_line"]["text"])}</p>')
    return panel(f'<h3>{badge("pass", "held out")}Le Petit Prince, section 8</h3>{g}'
                 f'<p class="big">On a held-out audiobook, the forecast was worth about {round(ve["point_k"])} real volunteers, and at least '
                 f'{ve["claim_k"]}. It agreed with the classic sound test.</p>{dots_svg(ve["point_k"], ve["claim_k"])}'
                 f'<p class="small">Each dot is one volunteer scan; a new story usually needs 12-15. Volunteer-equivalent {ve["point_k"]:.1f}, '
                 f'lower bound {ve["lower_k"]:.2f}, {s8["listeners"]} listeners.</p>'
                 f'<p>What it had to beat: word rate and the sound envelope (beaten in {k} of {len(beats)} comparisons). In language '
                 f'areas it beat a simple word-rate model in {wr["k_tribe_higher"]} of {wr["n"]} listeners.</p>'
                 f'<p class="small">{E(two)}</p>'
                 f'<p class="small">Le Petit Prince, section 8, held out: {S8_LABEL}. Baselines: {E(b["rule_timing"])} ({E(b["label"])}); '
                 '<code>results/section8.json</code>, <code>results/section8_baselines.json</code>.</p>')


WALL_CUTOFF_UTC = datetime.datetime(2026, 11, 1, 18, 0, tzinfo=datetime.timezone.utc)   # saturday_chain.sh: 42/42 predictions by then
WALL_NOT_RUN = "pre-registered, not run (GPU time ran out); no partial scoring"        # the chain's frozen not-run wording


def wall_html(d, now: datetime.datetime = None) -> str:
    """No relative days (ruling of Oct 30, 07:40 UTC, b): pending until the wall's frozen cutoff, then the frozen not-run wording."""
    if d is None:
        if (now or datetime.datetime.now(datetime.timezone.utc)) < WALL_CUTOFF_UTC:
            return pending("Story wall, Narratives and LPP", "Not scored yet; scored once, as pre-registered (D19).", SLOTS["story_wall"])
        return panel(f'<h3>{badge("info", "not run")}Story wall, Narratives and LPP</h3><p>Story wall: {E(WALL_NOT_RUN)}.</p>')
    w = d.get("wall", d); ln = (d.get("lines") or {}).get("line") or d.get("line")
    ok = bool(w.get("bar_pass"))
    guards = (d.get("lines") or {}).get("guards") or []
    return panel(f'<h3>{badge("pass" if ok else "fail", "bar met" if ok else "bar not reached")}Story wall, Narratives and LPP</h3>'
                 + (f"<p>{E(ln)}</p>" if ln else "") + f'<p class="small">{w["N"]} of {w["M"]} story families; the pre-set bar is two-thirds.'
                 + "".join(f" {E(g)}." for g in guards) + f' <code>{SLOTS["story_wall"]}</code>.</p>')


D20_PASS = ("With zero same-clip controls, stock TRIBE v2's language readout was stronger when the same healthy listeners heard a public "
            "story intact than when they heard its sentences shuffled (Narratives `notthefall`, N listeners, 9.1 minutes = 365 TRs per "
            "version; the order of their two runs is not public).")                     # PREREG D20's pass line, word for word


D20_CAVEAT = ("Each version also had its own TRIBE forecast, so a pass does not separate a change in the listeners' brains from a change in "
              "TRIBE's forecast, and the run order is not public, so it cannot be separated from an order or repeat-listening effect.")
PREREG_URL = f"{REPO_URL}/blob/main/PREREG.md"
PREREG_NOTE = ("Frozen pre-registration (D0-D27). Lines it quotes in its Never lists, and lines replaced by D18 and D23, are not claims.")


def d20_status(d) -> dict:
    """notthefall's ending (rulings of Oct 29, 18:15 IST item 3 and 17:35 UTC items 2-3), from results/d20_notthefall.json (the derived
    summary of the frozen outputs, scripts/make_d20_summary.py): a pass shows PREREG D20's pass line word for word with N filled, the two
    PREREG caveats and the secondary in its fail form as small print, and the stand-ins' row when the re-read matched the frozen run; any
    other ending leaves the D20 panel and its pair off the page (docs/captions.md: "the panel is left out")."""
    if d is None:
        return {"status": None}
    t = d["primary"]
    out = {"status": t["status"], "n": t["n"]}
    if t["status"] == "pass":
        out["line"] = D20_PASS.replace("N listeners", f"{t['n']} listeners")
        out["small"] = f"{D20_CAVEAT} {d['secondary']['line']}"
        b = d.get("baselines") or {}
        if b.get("matches_frozen"):
            out["baseline"] = (f"Exploratory, same test, language areas: word rate alone p = {b['word_rate_p']:.2f}, sound envelope alone "
                               f"p = {b['envelope_p']:.2f} ({b['n']} listeners, {d['minutes']} minutes = {d['trs_per_version']} TRs per version).")
    return out


def d20_html(d) -> str:
    if d is None:
        return pending("Real listeners, intact versus scrambled (Narratives notthefall)", "Shown as a result only if it passes its pre-set "
                       "bar.", SLOTS["d20"])
    st = d20_status(d)
    if st["status"] != "pass":
        return ""
    line = E(st["line"].replace("`", ""))                                     # the words as PREREG has them, without its code marks
    chip = GHOST_CHIPS["d20"]
    return panel(f'<h3>{badge("pass", "pass")}notthefall: real listeners, intact versus sentences shuffled</h3><p class="big">{line}</p>'
                 f'<p class="small">{E(st["small"])}</p>' + (f'<p>{E(st["baseline"])}</p>' if st.get("baseline") else "")
                 + f'<p class="small">{E(chip["short"])}: {E(chip["long"])}. <a href="{PREREG_URL}">PREREG.md</a>: {E(PREREG_NOTE)} '
                 f'<code>{SLOTS["d20"]}</code>.</p>')


FILM_FOOT = "Emo-FilM is in neither TRIBE v2's training list nor its evaluation list: {N} real viewers of the intact film."
FILM_GHOST_CAPTION = "Visual areas: TRIBE forecast (gray), {N} viewers (colour), intact film"          # ruling of Oct 29, 18:15 IST, item 4
GHOST_CHIPS = {"d20": {"short": "notthefall: rule 340b74b", "long": "scoring rule committed before these scans were read · 340b74b"}}
GHOST_COLOURS = {"group": ACCENT, "five": "#b4501a", "one": "#3d6b2e"}


def ghost_svg(doc: dict, keys=("group", "one"), title: str = "") -> str:
    """docs/rules/display_rule_ghostline.md: the gray dashed forecast in full first, then the chip, then the coloured group line, then
    the thin single line, in that DOM order; no smoothing; gaps (null) stay gaps; optional face ticks under the lines."""
    t, L = doc["time_s"], doc["lines"]
    W, H, l, r, top, b = 640, 250, 34, 12, 40, 30
    vals = [v for k in ("forecast",) + tuple(keys) for v in L[k]["values"] if v is not None]
    lo, hi = min(vals), max(vals); pad = (hi - lo) * 0.06 or 1
    lo, hi = lo - pad, hi + pad
    X = lambda x: l + (W - l - r) * (x - t[0]) / ((t[-1] - t[0]) or 1)
    Y = lambda y: top + (H - top - b) * (hi - y) / (hi - lo)

    def path(v):
        segs, cur = [], []
        for x, y in zip(t, v):
            if y is None:
                segs.append(cur); cur = []
            else:
                cur.append(f"{X(x):.1f},{Y(y):.1f}")
        return [c for c in segs + [cur] if len(c) > 1]
    P = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{E(title or doc.get("title", "ghost line"))}">']
    for tick in doc.get("ticks") or []:                                              # step 1, with the forecast
        P.append(f'<line x1="{X(tick[0]):.1f}" y1="{H - b}" x2="{X(tick[0]):.1f}" y2="{H - b - 14 * tick[1]:.1f}" stroke="#9aa3ad"/>')
    for seg in path(L["forecast"]["values"]):
        P.append(f'<polyline class="g-forecast" fill="none" stroke="#8a929b" stroke-width="2.4" stroke-dasharray="7 5" points="{" ".join(seg)}"/>')
    P.append(f'<text x="{l}" y="16" font-size="14" fill="{INK}" font-weight="600">{E(doc["chip"]["short"])}</text>')
    for k in keys:
        thin = k == "one"
        for seg in path(L[k]["values"]):
            P.append(f'<polyline class="g-{k}" fill="none" stroke="{GHOST_COLOURS.get(k, ACCENT)}" stroke-width="{1.2 if thin else 2.6}" '
                     f'points="{" ".join(seg)}"/>')
    x0 = l
    for k in ("forecast",) + tuple(keys):
        lab = L[k]["label"]
        col = "#8a929b" if k == "forecast" else GHOST_COLOURS.get(k, ACCENT)
        P.append(f'<rect x="{x0}" y="{top - 14}" width="14" height="4" fill="{col}"/><text x="{x0 + 18}" y="{top - 9}" font-size="12" '
                 f'fill="{MUTED}">{E(lab)}</text>')
        x0 += 30 + 6.2 * len(lab)
    return "".join(P) + f'<text x="{l}" y="{H - 8}" font-size="12" fill="{MUTED}">seconds of scan</text></svg>'


def film_score_html(fs: dict) -> str:
    """The intact film's volunteer-equivalent per gauge that ran, and the primary baseline comparisons."""
    rows = []
    for g, v in fs["isc"].items():
        ve = v["volunteer_equivalent"]; gz = fs["gauges"][g]
        rows.append(f'<tr><td>{E(g)}</td><td>about {round(ve["point_k"])}, at least {ve["claim_k"]}</td>'
                    f'<td>{gz["median_z"]:.2f} [{gz["ci95"][0]:.2f}, {gz["ci95"][1]:.2f}]</td></tr>')
    bl = fs.get("baselines") or {}; names = {"word_rate": "word rate", "envelope": "sound envelope", "brightness": "brightness", "motion": "motion"}
    prim = []
    for pair in bl.get("primary", []):
        net, base = pair.split(":"); r = bl.get(net)
        if net in ("AUD", "HEARING") and base == "envelope":                      # ruling of Oct 29, 18:15 IST, item 4: the headline
            continue                                                              # never says hearing beats the sound envelope
        if isinstance(r, dict) and base in r:
            prim.append(f"{net} vs {names.get(base, base)}: {r[base]['verdict']}")
    return (f'<details><summary>Every gauge that ran</summary><div class="tbl"><table><tr><th>Gauge</th><th>Worth (real viewers)</th>'
            f'<th>Median z [95% CI]</th></tr>{"".join(rows)}</table></div></details>'
            f'<p>What it had to beat: brightness, motion, sound envelope and word rate ({E("; ".join(prim) or "comparisons pending")}).</p>')


def film_result_html(fm, fs, ghost: dict = None) -> str:
    if fm is None:
        return pending("Film face test, 28 real viewers", "Sunday, scored once at the logged onsets. If the edit check fails or a gate "
                       "stops it, it is NOT RUN and the intact film is scored alone.", f'{SLOTS["face_map"]} and {SLOTS["film_score"]}')
    o = fm["outcome"]; N = fm.get("viewers"); b1 = fm.get("B1") or {}
    k, p = b1.get("k_above_0"), b1.get("wilcoxon_p")
    if o == "PASS":
        txt = (f"SAM filled the faces it masked; TRIBE forecast the change; that forecast lined up with where {N} real viewers' brains "
               f"tracked faces in the intact film ({k} of {N} viewers, p = {pnum(p)}), and more than for the control.")
        if fm.get("caption") == "carried by visual-network parcels":
            txt += " Most of the match comes from visual-network areas."
    elif o == "FAIL not face-specific":
        txt = (f"TRIBE's forecast lined up with where {N} real viewers' brains tracked faces ({k} of {N}), but no better than its forecast "
               "for the control: not specific to faces.")
    elif o == "FAIL":
        txt = (f"TRIBE's forecast of the filled faces did not line up with where {N} real viewers' brains tracked faces ({k} of {N} viewers, "
               f"p = {pnum(p)}). We show it because we said we would.")
    elif o == "UNINFORMATIVE":
        txt = f"The {N} viewers' own face maps did not agree with each other, so this film cannot test the forecast."
    elif o == "NOT RUN":
        txt = (f"The face test did not run: {fm['why']}. It was set to run once, by rules written before any brain data was read.")
    else:
        raise ValueError(f"unknown face-map outcome {o!r}")
    kind = "pass" if o == "PASS" else "info" if o == "NOT RUN" else "fail"
    body = f'<h3>{badge(kind, o)}Film face test</h3><p class="big">{E(txt)}</p>'
    if fs is not None:
        n = fs["viewers"]["scored"]
        body += f'<h3>The intact film, {n} real viewers</h3>' + film_score_html(fs)
        N = N or n
    elif o == "NOT RUN":
        body += f'<p class="small">The intact film\'s score fills from <code>{SLOTS["film_score"]}</code> when it lands.</p>'
    if ghost and N:
        body += (f'<div class="tbl">{ghost_svg(ghost, ("group", "one"))}</div><p class="small">{E(FILM_GHOST_CAPTION.format(N=N))}. '
                 f'{E(ghost["chip"]["short"])}: {E(ghost["chip"]["long"])}.</p>')
    if N:
        body += f'<p class="small">{E(FILM_FOOT.format(N=N))}</p>'
    return panel(body, id="face-test")


def film_status(fm) -> dict:
    """The Film station's one status line, read from the face-test result slot (ruling of Oct 30, 07:40 UTC, c)."""
    if fm is None:
        return {"text": "Face test: not scored yet.", "href": None}
    if fm["outcome"] == "NOT RUN":
        return {"text": f"The face test did not run: {fm['why']}. This station shows the edit it would have used.", "href": None}
    return {"text": f"Face test: {fm['outcome']}.", "href": "#face-test"}


# ---- page -------------------------------------------------------------------------------------------------------------------------
def build(inline: bool = False) -> str:
    sl = {k: load(v) for k, v in SLOTS.items()}
    media = load("demo/media/MEDIA.json"); spark = load("results/edit_check_spark_open.json")
    sts = studies(sl["demo_requests"], run_list(sl["demo_runs"]))
    picker, pick_css = picker_html(sts, sl["demo_runs"])
    P = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
         f'<title>StillHere Pilot</title><style>{CSS}{pick_css}</style></head><body><main>']
    P.append('<h1>StillHere Pilot</h1><p class="lead">Labs study people who can\'t respond after brain injury by playing them stories in the '
             'scanner. But every new story needs its own 12-15 healthy volunteers scanned first, in that language. For a new story, StillHere '
             'gives labs a first reading with zero volunteer scans, measured against each listener\'s own brain. It has been tested only in '
             'healthy volunteers.</p><p class="small">1 in 4 brain-injured patients who could not follow commands at the bedside showed '
             'command-following brain activity on fMRI or EEG (Bodien et al., NEJM 2024).</p>'
             f'<p class="small">{badge("info", "replay only")}This page replays recorded runs: it calls no model, needs no key, and every number '
             f'comes from a JSON file in <code>results/</code>. Research tool for labs. {E(ND)}</p>')
    P.append('<nav><a href="#pick">1 Pick a study</a><a href="#plan">2 Plan</a><a href="#check">3 Check</a><a href="#film">4 Film edit</a>'
             '<a href="#forecast">5 Forecast</a><a href="#held">6 Did it hold up?</a>'
             '<a href="#credits">Credits</a></nav>')
    P.append(f'<h2 id="pick">1. Pick a demo study</h2><p class="muted">A researcher types the study they want. Each request below was '
             f'labelled "demo, not scored". Round 1 is five requests as first written, run once live on every backend; round 2 rewords the '
             f'two that the default planner declined in round 1, written after seeing round 1. Both rounds are shown.</p>{picker}')
    P.append(f'<p class="beat">{E(planner_line(planner_rates()))}</p><p class="small">Planner v2, 62 held-out requests, three repeats; '
             '<code>results/planner_v2_eval.json</code> (exploratory).</p>')
    P.append(f'<h2 id="check">3. Check the stimulus before anyone is scanned</h2>{conditions_html(load("results/conditions_manipulation_check.json"))}')
    P.append(f'<h2 id="film">4. The same loop on video: the film edit</h2>{film_html(media, spark, sl["glimmer"])}')
    P.append(f'<h2 id="forecast">5. The forecast</h2>{forecast_html(load("results/live_tribe_librivox.json"), sl["live_conditions"])}')
    P.append('<h2 id="held">6. Did the forecasts hold up?</h2><p class="muted">Each forecast against real scans it never saw. Results land one '
             'at a time; a slot without its file says so.</p>')
    P.append(lpp_html(load("results/section8.json"), load("results/section8_baselines.json"), load("results/ghostline_lpp_s8.json")))
    P.append(wall_html(sl["story_wall"]) + d20_html(sl["d20"])
             + film_result_html(sl["face_map"], sl["film_score"], load("results/ghostline_film.json")))
    P.append(f'<p class="small" id="sedation">Sedation case study (not in the video): <a href="{SEDATION_URL}">the write-up, '
             'docs/writeup_sedation.md</a>.</p>')
    P.append('<footer id="credits"><p><b>A first healthy reference for a new test, with zero scans, and evidence for how far to trust it.</b></p><ul>'
             "<li>TRIBE v2 (Meta FAIR, CC BY-NC 4.0, not included in this repository) was trained on Courtois NeuroMod, BoldMoments, Lebel2023 "
             "and Wen2017 (25 participants) and evaluated by its authors on NNDb, LPP, Narratives and HCP (d'Ascoli et al. 2026, arXiv "
             "2605.04326, Table 1). LPP and Narratives are therefore among TRIBE v2's own evaluation sets. Emo-FilM, the Blender films and "
             "the sedation data (ds003171) are in neither list.</li>"
             "<li>SAM 3.1 through Meta's Model API, decoded with meta-sam-parser under Meta's SAM License; we acknowledge the use of SAM in "
             "this work.</li>"
             "<li>Data: Emo-FilM (Morgenroth et al. 2025; OpenNeuro ds004892 and ds004872, CC0), Le Petit Prince (ds003643), Narratives "
             "(ds002345), LibriVox (public domain), Tears of Steel (Blender Foundation, CC BY 3.0).</li>"
             f"<li>{E(CREDIT)}</li>"
             "<li>Research tool for labs. Tested only in healthy volunteers and public data. Not a diagnosis or medical advice.</li>"
             f"<li>{E(DISPLAY_NOTE)}</li>"
             "<li>Built with <code>scripts/build_pilot_page.py</code> from the JSON files named on each panel; media by "
             "<code>scripts/build_pilot_media.py</code> (<code>demo/media/MEDIA.json</code>).</li></ul></footer></main></body></html>")
    page = "".join(P)
    check(page)
    return inline_media(page) if inline else page


def text_of(page: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<style>.*?</style>", " ", page, flags=re.S)))


def check(page: str) -> None:
    """The build refuses banned wording (docs/captions.md), em dashes, scripts and remote resources."""
    low = text_of(page).lower()
    for w in BANNED:
        assert not re.search(r"\b" + re.escape(w) + r"\b", low), f"banned wording on the page: {w!r}"
    assert chr(0x2014) not in page, "em dash on the page"
    assert "<script" not in page.lower() and not re.search(r"""src=["']?(https?:)?//""", page), "script or remote resource"
    for m in re.finditer(r"""<(\w+)[^>]*?\s(\w+)=["']?((?:https?:)?//[^"'\s>]*)""", page):
        assert m.group(1).lower() == "a" and m.group(2) == "href" and m.group(3).startswith(REPO_URL + "/"), \
            f"remote link other than an anchor to the repository: {m.group(3)}"


def inline_media(page: str) -> str:
    def data(m):
        p = MEDIA / m.group(2); mime = "audio/mpeg" if p.suffix == ".mp3" else "image/jpeg"
        return f'{m.group(1)}="data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"'
    return re.sub(r'(src)="media/([^"]+)"', data, page)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--single"); a = ap.parse_args()
    OUT.write_text(build()); print(f"{OUT.relative_to(REPO)} {OUT.stat().st_size} bytes")
    if a.single:
        Path(a.single).write_text(build(inline=True)); print(f"{a.single} {Path(a.single).stat().st_size} bytes (media inlined)")
