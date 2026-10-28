"""Builds demo/index.html, the StillHere Pilot page (PLAN §7, reshaped for the Pilot direction), from JSON only: results/gate1_real.json
(frozen Gate 1 bars, group statistics only), results/section8.json (held-out LPP section 8), stillhere/validation_table.json (what can be
planned today), results/conditions_manipulation_check.json, results/diarization_validation.json, results/live_tribe_librivox.json,
demo/lines.json (PREREG §7 lines word for word, the planner examples, the section-8 label) and the Spark desk's REPLAY file. One
self-contained file: inline CSS and SVG, no scripts, fonts or network. The sedation case study carries only frozen group statistics: counts
per state and the desk's licensed lines (k of N with k*, HL and CI, p, fallback lines, lag marks, the D25 timing note); no per-volunteer
value, subject ID, timecourse or curve of the clip. Planner output that does not exist yet is labelled "planner v0 in progress" and is
never invented. usage: python scripts/build_demo_page.py"""
import html
import json
from pathlib import Path

from stillhere import desk, frozen

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "demo/index.html"
STATES = ("awake", "light", "deep", "recovery")
COL = {"HEARING": "#1d6fb8", "LANGUAGE": "#b4501a"}            # both >= 4.5:1 on white (tests check the contrast)
INK, MUTED, BAND, RULE = "#1b1f24", "#4a5560", "#eef0f3", "#9aa3ad"
E = html.escape


def load(name: str) -> dict:
    return json.loads((REPO / name).read_text())


def f2(x: float) -> str:
    return f"{x:.2f}"


def sgn(x: float) -> str:
    return f"{x:+.2f}"


def minutes(trs: int, tr_s: float) -> str:
    return f"{trs * tr_s / 60:.1f}"


def fill(t: str, v: dict) -> str:
    return t.format(**v)


def line_values(s8: dict, lines: dict) -> dict:
    ve, hy = s8["volunteer_equivalent_language"], {h["k"]: h for h in s8["hybrid_language"]}
    ws = max(s8["hybrid_wrong_story_language"], key=lambda w: (w["detect_pct"], -w["k"]))
    m3 = s8["m3_wrong_story"]
    return {"trs": s8["trs"], "minutes": minutes(s8["trs"], s8["tr_s"]), "listeners": s8["listeners"], "limit_pct": lines["limit_pct"],
            "k_about": round(ve["point_k"]), "k_point": f"{ve['point_k']:.1f}", "k_lower": f"{ve['lower_k']:.1f}", "k_claim": ve["claim_k"],
            "dz1": sgn(hy[1]["dz"]), "dz1_lo": sgn(hy[1]["dz_ci95"][0]), "dz1_hi": sgn(hy[1]["dz_ci95"][1]),
            "dz2": sgn(hy[2]["dz"]), "dz2_lo": sgn(hy[2]["dz_ci95"][0]), "dz2_hi": sgn(hy[2]["dz_ci95"][1]),
            "ws_n": round(ws["detect_pct"] * s8["listeners"] / 100), "ws_pct": f"{ws['detect_pct']:.0f}", "ws_k": ws["k"],
            "m3_n": m3["language"]["n"], "m3_h_pct": f"{m3['hearing']['pct']:.1f}", "m3_h_n": m3["hearing"]["detected"],
            "m3_l_pct": f"{m3['language']['pct']:.1f}", "m3_l_n": m3["language"]["detected"]}


# ---- charts (inline SVG) -------------------------------------------------------------------------------------------------
def curve_svg(s8: dict) -> str:
    W, H, l, r, t, b = 520, 330, 56, 16, 16, 52
    kmax, zmax = 16, 2.8
    x = lambda k: l + k / kmax * (W - l - r); y = lambda z: t + (zmax - z) / zmax * (H - t - b)
    pts = s8["isc_curve_language"]; ve = s8["volunteer_equivalent_language"]
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="curve-t"><title id="curve-t">ISC with k volunteers, and TRIBE alone</title>']
    for z in (0, 0.5, 1, 1.5, 2, 2.5):
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{y(z):.1f}" y2="{y(z):.1f}" stroke="{BAND}"/><text x="{l - 8}" y="{y(z) + 4:.1f}" text-anchor="end" class="tick">{z:g}</text>')
    for k in (1, 3, 6, 9, 12, 15):
        out.append(f'<text x="{x(k):.1f}" y="{H - b + 20}" text-anchor="middle" class="tick">{k}</text>')
    out.append(f'<text x="{(l + W - r) / 2:.0f}" y="{H - 10}" text-anchor="middle" class="axis">k, other listeners in the reference</text>')
    out.append(f'<text x="14" y="{(t + H - b) / 2:.0f}" transform="rotate(-90 14 {(t + H - b) / 2:.0f})" text-anchor="middle" class="axis">LANGUAGE z</text>')
    out.append(f'<polyline fill="none" stroke="{MUTED}" stroke-width="2.5" points="{" ".join(f"{x(p["k"]):.1f},{y(p["z"]):.1f}" for p in pts)}"/>')
    out += [f'<circle cx="{x(p["k"]):.1f}" cy="{y(p["z"]):.1f}" r="4" fill="{MUTED}"/>' for p in pts]
    tx, ty = x(ve["point_k"]), y(ve["z"])
    out.append(f'<line x1="{tx:.1f}" x2="{tx:.1f}" y1="{y(ve["ci95"][0]):.1f}" y2="{y(ve["ci95"][1]):.1f}" stroke="{COL["LANGUAGE"]}" stroke-width="2.5"/>')
    out.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="7" fill="{COL["LANGUAGE"]}"/><text x="{tx + 12:.1f}" y="{ty + 18:.1f}" class="legend">TRIBE alone</text>')
    out.append(f'<text x="{x(pts[-1]["k"]) - 4:.1f}" y="{y(pts[-1]["z"]) - 10:.1f}" text-anchor="end" class="legend">ISC(k)</text>')
    return "".join(out) + "</svg>"


def hybrid_svg(s8: dict) -> str:
    W, H, l, r, t, b = 520, 330, 56, 16, 16, 52
    hy = s8["hybrid_language"]; ks = [h["k"] for h in hy]; zmax = 2.4
    x = lambda k: l + 30 + (k - ks[0]) / (ks[-1] - ks[0]) * (W - l - r - 60); y = lambda z: t + (zmax - z) / zmax * (H - t - b)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="hyb-t"><title id="hyb-t">k controls alone and TRIBE plus k controls</title>']
    for z in (0, 0.5, 1, 1.5, 2):
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{y(z):.1f}" y2="{y(z):.1f}" stroke="{BAND}"/><text x="{l - 8}" y="{y(z) + 4:.1f}" text-anchor="end" class="tick">{z:g}</text>')
    for k in ks:
        out.append(f'<text x="{x(k):.1f}" y="{H - b + 20}" text-anchor="middle" class="tick">{k}</text>')
    out.append(f'<text x="{(l + W - r) / 2:.0f}" y="{H - 10}" text-anchor="middle" class="axis">k, real controls</text>')
    out.append(f'<text x="14" y="{(t + H - b) / 2:.0f}" transform="rotate(-90 14 {(t + H - b) / 2:.0f})" text-anchor="middle" class="axis">LANGUAGE z</text>')
    for key, colr, lab in (("alone_z", MUTED, "k controls alone"), ("hybrid_z", COL["HEARING"], "TRIBE plus the same k controls")):
        out.append(f'<polyline fill="none" stroke="{colr}" stroke-width="2.5" points="{" ".join(f"{x(h["k"]):.1f},{y(h[key]):.1f}" for h in hy)}"/>')
        out += [f'<circle cx="{x(h["k"]):.1f}" cy="{y(h[key]):.1f}" r="4.5" fill="{colr}"/>' for h in hy]
        ly = t + (14 if key == "hybrid_z" else 36)
        out.append(f'<line x1="{l + 14}" x2="{l + 40}" y1="{ly}" y2="{ly}" stroke="{colr}" stroke-width="3"/><text x="{l + 48}" y="{ly + 4}" class="legend">{lab}</text>')
    return "".join(out) + "</svg>"


def tier_svg() -> str:
    W, H = 1000, 200
    BW = 260
    boxes = [(10, "Kaggle, private kernels", ["BOLD, TRIBE predictions and the", "privately shared stimulus stay here", "(never downloaded, never shown)"]),
             (370, "Laptop", ["tier-D numbers only: counts,", "z and p values, CIs, hashes", "(this page reads only these)"]),
             (730, "Meta Model API (Muse Spark)", ["typed egress only: numbers, enums,", "SHA-256-allowlisted public text;", "-contributor models refused"])]
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="tier-t"><title id="tier-t">Where the data go</title>',
           '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
           f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>']
    for x0, head, rows in boxes:
        out.append(f'<rect x="{x0}" y="30" width="{BW}" height="140" rx="10" fill="#ffffff" stroke="{RULE}" stroke-width="1.5"/>')
        out.append(f'<text x="{x0 + BW / 2:.0f}" y="62" text-anchor="middle" class="boxhead">{E(head)}</text>')
        out += [f'<text x="{x0 + BW / 2:.0f}" y="{94 + i * 24}" text-anchor="middle" class="boxrow">{E(rw)}</text>' for i, rw in enumerate(rows)]
    for x1, x2, lab in ((10 + BW, 370, "tier-D only"), (370 + BW, 730, "typed egress")):
        out.append(f'<line x1="{x1 + 6}" x2="{x2 - 6}" y1="104" y2="104" stroke="{MUTED}" stroke-width="2" marker-end="url(#arr)"/>')
        out.append(f'<text x="{(x1 + x2) / 2:.0f}" y="92" text-anchor="middle" class="tick">{lab}</text>')
    return "".join(out) + "</svg>"


# ---- page ---------------------------------------------------------------------------------------------------------------
IN_PROGRESS = "planner v0 in progress"
EXPL = "exploratory"
SCANS = (("lpp", "LPP section 8"), ("wall", "Story wall"), ("d20", "notthefall D20"), ("film", "Blender film / Emo-FilM"), ("sedation", "Sedation case study"))
PLACEHOLDERS = {"wall": "runs Thu Oct 29", "d20": "scored Thu Oct 29 after the wall", "film": "rule committed before any brain data is read; results pending"}

CSS = f"""
*{{box-sizing:border-box}} body{{margin:0;background:#ffffff;color:{INK};font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:28px 20px 48px}} h1{{font-size:2rem;margin:0 0 4px}} h2{{font-size:1.3rem;margin:40px 0 8px}}
h3{{font-size:1.08rem;margin:26px 0 6px}} .sub{{color:{MUTED};margin:0 0 8px}} .badge{{display:inline-block;font-size:.8rem;font-weight:600;letter-spacing:.04em;
border:1px solid {RULE};border-radius:999px;padding:2px 10px;color:{MUTED};margin-right:6px}} figure{{margin:16px 0}} figcaption,.cap{{color:{MUTED};font-size:.92rem}}
svg{{width:100%;height:auto;display:block}} .tick{{font-size:12px;fill:{MUTED}}} .axis{{font-size:13px;fill:{MUTED}}} .state{{font-size:15px;font-weight:600;fill:{INK}}}
.legend{{font-size:13px;fill:{INK}}} .boxhead{{font-size:16px;font-weight:700;fill:{INK}}} .boxrow{{font-size:13px;fill:{MUTED}}}
.lines{{list-style:none;padding:0;margin:12px 0}} .lines li{{border-left:4px solid {RULE};padding:6px 12px;margin:10px 0}}
.lines li.fail{{border-color:{COL["LANGUAGE"]}}} .lines li.pass{{border-color:{COL["HEARING"]}}} q{{quotes:none}} blockquote{{margin:12px 0;padding:10px 14px;background:{BAND};border-radius:8px}}
.src{{color:{MUTED};font-size:.88rem;display:block;margin-top:4px}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
.card{{border:1px solid {RULE};border-radius:10px;padding:14px 16px;font-size:.95rem}} .card p{{margin:6px 0}} .card .head{{font-weight:600}}
table{{border-collapse:collapse;width:100%;font-size:.95rem}} td,th{{text-align:left;padding:6px 8px;border-bottom:1px solid {BAND};vertical-align:top}} th{{color:{MUTED};font-weight:600}}
.tabs{{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 4px;padding:0;list-style:none}} .tabs a{{display:inline-block;padding:4px 12px;border:1px solid {RULE};
border-radius:999px;color:{INK};text-decoration:none;font-size:.92rem}} .pending{{border:1px dashed {RULE};border-radius:10px;padding:12px 16px;color:{MUTED}}}
.res-pass{{color:{COL["HEARING"]};font-weight:600}} .res-fail{{color:{COL["LANGUAGE"]};font-weight:600}} .tablewrap{{overflow-x:auto}}
footer{{margin-top:44px;padding-top:14px;border-top:1px solid {RULE};color:{MUTED};font-size:.9rem}}
@media (max-width:720px){{.grid{{grid-template-columns:1fr}} h1{{font-size:1.6rem}} main{{padding:20px 14px 40px}}
figure.wide,figure.mid{{overflow-x:auto;-webkit-overflow-scrolling:touch}} figure.wide svg{{min-width:760px}} figure.mid svg{{min-width:460px}}
figure.wide figcaption,figure.mid figcaption{{position:sticky;left:0;max-width:calc(100vw - 28px)}}}}
"""


def flow_svg() -> str:
    """The Pilot flow as five boxes: neutral labels, no numbers."""
    W, H, n = 1000, 120, 5
    steps = ("Researcher types a study", "Planner (Muse Spark)", "Stimulus conditions", "Manipulation check", "TRIBE v2 forecast")
    bw, gap = 172, (1000 - 5 * 172) / 4
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="flow-t"><title id="flow-t">StillHere Pilot flow</title>',
           '<defs><marker id="fl" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
           f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>']
    for k, lab in enumerate(steps):
        x0 = k * (bw + gap)
        out.append(f'<rect x="{x0:.1f}" y="24" width="{bw}" height="72" rx="10" fill="#ffffff" stroke="{RULE}" stroke-width="1.5"/>')
        words = lab.split(" "); mid = (len(words) + 1) // 2
        for r, part in enumerate((" ".join(words[:mid]), " ".join(words[mid:]))):
            out.append(f'<text x="{x0 + bw / 2:.1f}" y="{56 + r * 20}" text-anchor="middle" class="boxrow">{E(part)}</text>')
        if k < n - 1:
            out.append(f'<line x1="{x0 + bw + 4:.1f}" x2="{x0 + bw + gap - 4:.1f}" y1="60" y2="60" stroke="{MUTED}" stroke-width="2" marker-end="url(#fl)"/>')
    return "".join(out) + "</svg>"


def s8(text: str, lines: dict) -> str:
    """Append the section-8 label to any text that mentions section 8 (held-out LPP English section 8)."""
    return f'{text} ({lines["s8_label"]})' if "section 8" in text and lines["s8_label"] not in text else text


def planned_today(vt: dict) -> list:
    """(dimension, id, status, evidence) for every validation-table entry, in table order."""
    return [(k, e["id"], e["status"], e["evidence"]) for k, v in vt.items() if isinstance(v, list) for e in v]


def expected_decline(vt: dict, ex: dict) -> tuple:
    """The decline the planner must give for one example request, derived from the validation table alone (not model output)."""
    ids = {e["id"]: e for e in vt[ex["dimension"]]}
    if ex["dimension"] == "population" and ex["asked"] not in ids:
        near = [e["id"] for e in vt["population"] if e["status"] == "validated"]
        return "population_not_validated", near[0] if near else None
    if ex["dimension"] == "analysis" and ids.get(ex["asked"], {}).get("status") == "not_offered":
        near = [e["id"] for e in vt["analysis"] if e["status"] == "validated"]
        return "analysis_not_offered", near[0] if near else None
    raise ValueError(f"no decline derivable for {ex}")


def plan_html(vt: dict, lines: dict, cm: dict, dz: dict, lt: dict) -> str:
    """Section 1: the planner walkthrough from real artefacts; nothing a model has not produced is shown as model output."""
    P = [f'<p class="cap"><span class="badge">{E(IN_PROGRESS)}</span>Planner v0 is being built by Muse Code. Shown here: what it may plan '
         'today (its enums are rebuilt from this table on every request), the declines this table forces, and the real artefacts the planner '
         'calls. No planner model output is shown yet.</p>']
    P.append('<h3>What can be planned today (stillhere/validation_table.json)</h3><div class="tablewrap"><table><tr><th>Field</th><th>Value</th>'
             '<th>Status</th><th>Evidence</th></tr>')
    P += [f'<tr><td>{E(k)}</td><td>{E(i)}</td><td>{E(st)}</td><td>{E(s8(ev, lines))}</td></tr>' for k, i, st, ev in planned_today(vt)]
    P.append('</table></div>')
    P.append(f'<h3>Declines the table forces ({E(IN_PROGRESS)}: derived from the table, not model output)</h3><ul class="lines">')
    for ex in lines["planner_examples"]:
        code, near = expected_decline(vt, ex)
        P.append(f'<li class="fail"><q>{E(ex["request"])}</q> declines with <b>{E(code)}</b>; nearest validated option: {E(near)}.</li>')
    P.append('</ul>')
    c = cm["conditions"]
    P.append(f'<h3>Stimulus conditions and their manipulation check ({E(cm["label"])})</h3>'
             f'<p class="cap">Built from one public-domain LibriVox clip ({E(cm["source"])}), seed {cm["seed"]}; checked with Meta\'s Voice '
             'Transcribe. Words recovered and order are shares of the intact transcript.</p><div class="tablewrap"><table><tr><th>Condition</th>'
             '<th>Words recovered</th><th>Order vs expected</th><th>Result</th></tr>')
    for name, r in c.items():
        res = ('<span class="res-pass">pass</span>' if r["pass"] else '<span class="res-fail">fail</span>') if r["pass"] is not None else "no word check (reversed speech)"
        P.append(f'<tr><td>{E(name)}</td><td>{r["recall"]:.3f}</td><td>{r["order_vs_expected"]:.3f}</td><td>{res}</td></tr>')
    P.append('</table></div>')
    cr = dz["criterion"]
    P.append(f'<p class="cap">Diarisation validated before use as a gate ({E(dz["label"])}): {dz["n_labels"]} speaker labels (criterion '
             f'{cr["n_labels"]}), agreement {dz["agreement"]:.3f} (criterion at least {cr["agreement_min"]:.2f}), {dz["speaker_changes"]} speaker '
             f'changes against {dz["true_speaker_changes"]} true (tolerance {cr["changes_tolerance"]}): {"pass" if dz["pass"] else "fail"}.</p>')
    P.append(f'<h3>Forecast, live on the laptop</h3><div class="card"><p class="head">{E(lt["label"])}</p>'
             f'<p>Predicted auditory response strength {lt["predicted_aud_response_strength"]:.3f} over {lt["n_aud_parcels"]} auditory parcels; '
             f'{lt["seconds_predicted"]} s predicted in {lt["predict_s"]:.2f} s on a laptop GPU, peak {lt["peak_vram_gb"]:.2f} GB.</p>'
             f'<p>{E(lt["wording"])}</p></div>')
    return "".join(P)


def sedation_html(g: dict, fz) -> str:
    """The sedation case study: frozen group statistics only (counts per state and the desk's licensed lines)."""
    inc = g["included"]; mins = minutes(g["trs_per_state"], g["tr_s"])
    P = ['<p class="cap">Frozen group statistics only (PREREG §6 bars, Gate 1): no per-volunteer value, no subject ID, no timecourse. '
         'Included per state: ' + ", ".join(f'{st.capitalize()} {inc[st]}' for st in STATES) +
         f'. {g["trs_per_state"]} TRs = {mins} minutes per state.</p><ul class="lines">']
    for lid in desk.licensed_lines(fz):
        cls = "fail" if lid.endswith("fallback") or lid == "sedation_no_line" else "pass"
        P.append(f'<li class="{cls}">{E(desk.render(fz, lid))}</li>')
    P.append(f'<li>{E(desk.TIMING_NOTE)}</li></ul>')
    return "".join(P)


def lpp_html(s8d: dict, lines: dict, v: dict) -> str:
    """LPP section 8: the PREREG §7 lines word for word, the table and the two curves, each section-8 mention labelled."""
    st, s8m, ve = s8d["stage"], minutes(s8d["trs"], s8d["tr_s"]), s8d["volunteer_equivalent_language"]
    P = ['<ul class="lines">']
    for ln in lines["lpp"]:
        q, note = fill(ln["quote"], v), (fill(ln["note"], v) if ln["note"] else "PREREG §7")
        lab = f' ({lines["s8_label"]})' if ("section 8" in q or "section 8" in note) else ""
        P.append(f'<li><q>{E(q)}</q><span class="src">{E(note + lab)}</span></li>')
    P.append(f'</ul><p class="cap">M3 caveat (PREREG §7): {E(s8(fill(lines["m3"], v), lines))}</p>')
    head = s8(f'Held-out English section 8, {s8d["listeners"]} listeners, {s8d["trs"]} TRs = {s8m} minutes', lines)
    rows = (("LANGUAGE, real words", st["language_z"]), ("LANGUAGE, word-shuffled prediction", st["shuffled_z"]),
            ("Best word-level reference (language-model surprisal)", st["surprisal_z"]), ("HEARING", st["hearing_z"]),
            ("Aligned sound envelope", st["envelope_z"]))
    P.append(f'<div class="tablewrap"><table><tr><th>{E(head)}</th><th>z</th></tr>'
             + "".join(f"<tr><td>{E(a)}</td><td>{f2(z)}</td></tr>" for a, z in rows) + "</table></div>")
    cap1 = s8(f'ISC(k), held-out English section 8: brain-to-brain correlation with k other listeners as the reference. TRIBE alone, '
              f'z {f2(ve["z"])} [95% CI {f2(ve["ci95"][0])}, {f2(ve["ci95"][1])}], plotted at its point on the curve ({ve["point_k"]:.1f}; '
              f'lower 95% bound {ve["lower_k"]:.1f}). {s8d["trs"]} TRs = {s8m} minutes. M3 caveat applies.', lines)
    gains = "; ".join(f'k = {h["k"]} {sgn(h["dz"])} [{sgn(h["dz_ci95"][0])}, {sgn(h["dz_ci95"][1])}]' for h in s8d["hybrid_language"])
    cap2 = s8(f'k real controls alone, and TRIBE added as one more reference subject (hybrid rule fixed in advance), held-out English '
              f'section 8. Paired gain: {gains}. {s8d["trs"]} TRs = {s8m} minutes. M3 caveat applies.', lines)
    P.append(f'<div class="grid"><figure class="mid">{curve_svg(s8d)}<figcaption>{E(cap1)}</figcaption></figure>'
             f'<figure class="mid">{hybrid_svg(s8d)}<figcaption>{E(cap2)}</figcaption></figure></div>')
    return "".join(P)


def baselines_html(b: dict, lines: dict) -> str:
    """Section 8: TRIBE next to two cheap baselines (HRF x word rate, HRF x sound envelope), scored the same way; exploratory, with the
    rule-timing note next to the table."""
    head = s8(f'Cheap baselines, held-out English section 8, {b["145"]["n"]} listeners ({b["label"]}; {b["rule_timing"]})', lines)
    P = [f'<div class="tablewrap"><table><tr><th>{E(head)}</th><th>TRIBE z</th><th>Word rate z</th><th>TRIBE minus word rate</th>'
         f'<th>Envelope z</th><th>TRIBE minus envelope</th></tr>']
    for win, wlab in (("145", "145 TRs"), ("full", "full length")):
        for net, nlab in (("LANG", "language parcels"), ("AUD", "auditory parcels")):
            r = b[win][net]; m, vs = r["mean_z"], r["tribe_vs"]
            cell = lambda c: f'{sgn(c["hl"])} [{sgn(c["hl_ci95"][0])}, {sgn(c["hl_ci95"][1])}], {c["k_tribe_higher"]} of {c["n"]}'
            P.append(f'<tr><td>{E(wlab + ", " + nlab)}</td><td>{f2(m["tribe"])}</td><td>{f2(m["word_rate"])}</td><td>{E(cell(vs["word_rate"]))}</td>'
                     f'<td>{f2(m["envelope"])}</td><td>{E(cell(vs["envelope"]))}</td></tr>')
    P.append('</table></div><p class="cap">Hodges-Lehmann median of TRIBE minus the baseline with its 95% listener-bootstrap interval, and '
             'the number of listeners for whom TRIBE is higher. Same parcels, nulls and listeners for all three; no partialling.</p>')
    return "".join(P)


def build() -> str:
    g, s8d, lines = load("results/gate1_real.json"), load("results/section8.json"), load("demo/lines.json")
    vt, cm, dz, lt = load("stillhere/validation_table.json"), load("results/conditions_manipulation_check.json"), \
        load("results/diarization_validation.json"), load("results/live_tribe_librivox.json")
    fz = frozen.Frozen(g); v = line_values(s8d, lines)
    q = desk.public("q_language_fades"); card = desk.Desk(fz, replay=REPO / "replay/desk_real.jsonl").ask(q)
    P = []
    P.append(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
             f'<title>StillHere Pilot</title><style>{CSS}</style></head><body><main>')
    P.append('<h1>StillHere Pilot</h1><p class="sub">A researcher types the study they want to run. StillHere builds the stimulus conditions and '
             'their controls, Meta\'s Voice Transcribe checks them, and TRIBE v2 forecasts the healthy-adult brain response. Every forecast is '
             'shown next to real public scans.</p><p class="sub">Research tool for brain-imaging labs, built on Meta\'s TRIBE v2. Healthy '
             'volunteers, public data. Not a diagnosis or medical advice.</p>')
    P.append(f'<blockquote><q>{E(" ".join(lines["pitch"]))}</q><span class="src">PREREG §7, allowed now, first sentence as replaced by D18</span></blockquote>')
    P.append(f'<figure class="wide">{flow_svg()}</figure>')
    # 1. plan a study
    P.append(f'<h2 id="plan">1. Plan a study</h2>{plan_html(vt, lines, cm, dz, lt)}')
    # 2. against real scans
    P.append('<h2 id="scans">2. How the forecasts did against real scans</h2><ul class="tabs">'
             + "".join(f'<li><a href="#{k}">{E(t)}</a></li>' for k, t in SCANS) + '</ul>')
    for k, t in SCANS:
        P.append(f'<h3 id="{k}">{E(t)}</h3>')
        if k in PLACEHOLDERS:
            P.append(f'<p class="pending">{E(PLACEHOLDERS[k])}</p>')
        elif k == "lpp":
            P.append(f'<p class="cap">{E(s8("Public Le Petit Prince listeners, no same-story volunteers; English numbers from held-out section 8", lines))}.</p>')
            P.append(lpp_html(s8d, lines, v)); P.append(baselines_html(load("results/section8_baselines.json"), lines))
        elif k == "sedation":
            P.append(sedation_html(g, fz))
    # 3. desk, 4. tiers
    head, *body = card.text.split("\n")
    P.append(f'<h2 id="desk">3. Spark analysis desk</h2><p class="cap"><span class="badge">{E(card.badge)}</span>Question from the menu: '
             f'<q>{E(q.text.strip())}</q> Replayed from a recorded Muse Spark session; the desk licenses each line from the frozen bars.</p>')
    P.append(f'<div class="card"><p class="head">{E(head)}</p>' + "".join(f"<p>{E(b)}</p>" for b in body) + "</div>")
    P.append(f'<h2 id="tiers">4. Where the data go</h2><figure class="wide">{tier_svg()}</figure>')
    P.append('<footer><p>Research tool, not a diagnosis. Built with Llama.</p><p>TRIBE v2: Meta FAIR (d\'Ascoli et al. 2026), weights and code '
             'released under CC BY-NC 4.0. Data: OpenNeuro ds003643 (Le Petit Prince), ds003171 (propofol sedation, frozen group statistics '
             'only), ds002345 (Narratives) and ds004892 (Emo-FilM); LibriVox recordings (public domain); Tears of Steel (Blender Foundation, '
             'CC BY 3.0). Every number on this page comes from a JSON file in results/ or stillhere/; bars and wording are frozen in PREREG.md. '
             'New results are labelled exploratory.</p></footer></main></body></html>')
    return "".join(P)


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(build()); print(OUT.relative_to(REPO), len(OUT.read_text()), "bytes")
