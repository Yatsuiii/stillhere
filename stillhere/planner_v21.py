"""Study planner v2.1 (StillHere Pilot live page). Planner v2 (stillhere/planner.py) stays exactly as published and measured; v2.1
sits beside it and is NOT re-measured (badge: "v2.1 adds clip, condition, seed; not re-measured").

What v2.1 changes from v2 (cloud rulings of Oct 29, 18:15 IST, items 1 and 2):
1. Strings only for the model step: "Nearest tested option" (v2: "Nearest validated option"), "Forecast, not checked against scans"
   (v2: "Forecast, not validated"). Decline codes and table ids never reach a person raw: plain_reason() and plain_id() are the one
   mapping, and every card is built from it.
2. The StudySpec adds clip_id (the live menu, stillhere/live_menu.json, rebuilt on every request), condition (one of the conditions the
   spec lists) and seed (an integer for the sentence shuffle). A decline nulls all three, like every other field.
3. The model only plans: one json_schema call (plus v2's single re-prompt), no tool call. Our code turns the spec into a route by its
   stimulus: a live clip goes to the stations build -> check -> forecast for exactly that (clip_id, condition, seed); a stimulus that is
   not on the live menu (Le Petit Prince) ends on its evidence panel. run_stations() never shows an output made for another clip,
   condition or seed (the bug v2's model tool loop had: Le Petit Prince plans got the LibriVox clip's outputs).

The executors for build, check and forecast belong to the live page and its worker and are passed in; without one a station is
"pending", never a stand-in from another clip.
"""
import json
from pathlib import Path

from . import egress
from . import planner as v2

VERSION = "v2.1"
BADGE = "v2.1 adds clip, condition, seed; not re-measured"
MENU = Path(__file__).with_name("live_menu.json")
FIELDS = ("population", "language", "stimulus", "analysis", "brain_areas", "conditions", "decline_reason", "clip_id", "condition", "seed")
SEED_MAX = 2 ** 31 - 1
STATIONS = ("build", "check", "forecast")
MODELS = v2.MODELS

# ---- plain words: the one mapping from codes and table ids to what a person reads ------------------------------------------------
REASON_WORDS = {
    "population_not_validated": "that group of people is not on the tested menu",
    "language_not_validated": "that language is not on the tested menu",
    "brain_area_not_validated": "that brain area is not on the tested menu",
    "condition_not_validated": "that condition is not on the tested menu",
    "stimulus_not_validated": "that story or clip is not on the tested menu",
    "analysis_not_offered": "that analysis is not offered",
    "clinical_out_of_scope": "questions about patients or one person's scan are out of scope for this research tool",
}
ID_WORDS = {
    "healthy_adults": "healthy adults", "english": "English", "french": "French (group level only)",
    "auditory": "hearing areas", "language_network": "language network",
    "intact": "intact", "sentence_scramble": "sentences shuffled", "reversed": "played backwards", "word_scramble": "words shuffled",
    "librivox_audiobook": "LibriVox audiobook", "le_petit_prince": "Le Petit Prince", "blender_film": "Blender film",
    "forecast_tribe_stock": "stock TRIBE v2 forecast", "lag_search": "lag search",
}
AXIS_WORDS = {"population": "people", "language": "language", "brain_area": "brain area", "condition": "condition",
              "stimulus": "stimulus", "analysis": "analysis"}


def plain_reason(code: str) -> str:
    return REASON_WORDS.get(code, "the request is outside the tested menu")


def plain_id(i) -> str:
    if i is None:
        return "none"
    return ID_WORDS.get(i) or _clip_titles().get(i) or i.replace("_", " ")


def nearest_words(reason: str, enums: dict) -> str:
    axis, _, opt = v2.nearest_option(reason, enums).partition(" ")
    return f"{AXIS_WORDS.get(axis, axis)}: {plain_id(opt) if opt and not opt.startswith('(') else 'none offered'}"


# ---- the live menu -----------------------------------------------------------------------------------------------------------
def live_menu(path: Path | None = None) -> dict:
    return json.loads(Path(path or MENU).read_text())


def _clip_titles() -> dict:
    return {c["id"]: c["title"] for c in live_menu().get("clips", [])}


def load_enums(table: Path | None = None, menu: Path | None = None) -> dict:
    """v2's enums from the validation table, plus clip_id from the live menu (both rebuilt on every request) and live_condition: the
    manipulated conditions a live run forecasts against intact (the table's conditions other than intact)."""
    e = v2.load_enums(table)
    return e | {"clip_id": [c["id"] for c in live_menu(menu).get("clips", [])], "live_condition": [c for c in e["condition"] if c != "intact"]}


def menu_text(enums: dict | None = None) -> str:
    """The line above the input box: what a live run can use, built from the table and the live menu."""
    e = enums or load_enums()
    short = {c["id"]: c.get("short", c["title"]) for c in live_menu().get("clips", [])}
    clips = ", ".join(short[c] for c in e["clip_id"])
    conds = " or ".join(plain_id(c) for c in e["live_condition"])
    return (f"Live menu: {plain_id(e['population'][0])} listening in {plain_id('english')} to a LibriVox clip ({clips}; "
            f"40 s each), intact against {conds}; brain areas {', '.join(plain_id(b) for b in e['brain_area'])}; "
            f"analysis: {plain_id(e['analysis'][0])}. A Le Petit Prince plan ends on its held-out evidence panel. Anything else gets a "
            f"reason and the nearest tested option.")


# ---- schema and validator ----------------------------------------------------------------------------------------------------
def _nullable(vals) -> dict:
    return {"type": ["string", "null"], "enum": sorted(set(vals)) + [None]}


def study_schema(enums: dict) -> dict:
    s = v2.study_schema(enums)
    s["properties"] |= {"clip_id": _nullable(enums["clip_id"]), "condition": _nullable(enums["live_condition"]),
                        "seed": {"type": ["integer", "null"]}}
    s["required"] = list(FIELDS)
    return s


def validate_answer(ans, enums: dict) -> list:
    """v2's checks on the v2 fields, plus: clip_id on the live menu or null, condition a manipulated (non-intact) condition the spec
    lists, or null; seed an integer in [0, 2**31 - 1] or null. A decline only needs to be schema-valid (plan() nulls its fields)."""
    if not isinstance(ans, dict) or set(ans) != set(FIELDS):
        return ["schema_invalid"]
    seed = ans.get("seed")
    if (ans.get("clip_id") not in enums["clip_id"] + [None] or ans.get("condition") not in enums["live_condition"] + [None]
            or not (seed is None or (isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed <= SEED_MAX))):
        return ["schema_invalid"]
    codes = v2.validate_answer({k: ans[k] for k in FIELDS[:7]}, enums)
    if codes or ans["decline_reason"] != "none":
        return codes
    if ans["condition"] is not None and ans["condition"] not in ans["conditions"]:
        return ["decline_inconsistent"]
    return []


def decline_spec(reason: str) -> dict:
    return v2.decline_spec(reason) | {"clip_id": None, "condition": None, "seed": None}


# ---- routing: our code runs the stations from the spec ---------------------------------------------------------------------
def route(spec: dict, enums: dict | None = None) -> dict:
    """Where an accepted spec goes, by its stimulus. librivox_audiobook -> the live stations for one (clip_id, condition, seed), with
    stated defaults when the request named none; any other stimulus -> its evidence panel, with clip, condition and seed dropped."""
    if (spec.get("decline_reason") or "none") != "none" or not spec.get("stimulus"):
        return {"kind": "decline", "reason": spec.get("decline_reason")}
    e = enums or load_enums()
    menu = live_menu()
    if spec["stimulus"] == "librivox_audiobook" and e["clip_id"]:
        defaults = []
        clip = spec.get("clip_id")
        if clip is None:
            clip = e["clip_id"][0]
            defaults.append("clip")
        cond = spec.get("condition")
        if cond is None:
            cond = next(c for c in spec["conditions"] if c != "intact")          # an accept lists intact and at least one other
            defaults.append("condition")
        seed = spec.get("seed")
        if seed is None:
            seed = int(menu["default_seed"])
            defaults.append("seed")
        return {"kind": "live", "clip_id": clip, "condition": cond, "seed": seed, "defaults": defaults, "stations": list(STATIONS)}
    panel = menu.get("evidence_panels", {}).get(spec["stimulus"])
    return {"kind": "evidence", "stimulus": spec["stimulus"], "panel": panel["panel"] if panel else None,
            "results": panel["results"] if panel else None,
            "line": panel["line"] if panel else f"{plain_id(spec['stimulus'])} is not on the live menu; there is no live run for it."}


def _pending(rt: dict, station: str) -> dict:
    return {"status": "pending", "clip_id": rt["clip_id"], "condition": rt["condition"], "seed": rt["seed"],
            "note": f"{station}: computed by the live worker for this clip; nothing from another clip is shown"}


def run_stations(rt: dict, executors: dict | None = None) -> list:
    """Runs the route's stations in order with the given executors (station -> callable(route) -> dict). Every output must name the
    route's own clip_id, condition and seed; an output for anything else is refused, not shown. An evidence route returns its panel
    only; a decline returns nothing."""
    if rt["kind"] == "decline":
        return []
    if rt["kind"] == "evidence":
        return [{"station": "evidence", "output": {"panel": rt["panel"], "results": rt["results"], "line": rt["line"]}}]
    out, executors = [], executors or {}
    for st in rt["stations"]:
        fn = executors.get(st)
        res = fn(rt) if fn else _pending(rt, st)
        if any(res.get(k) != rt[k] for k in ("clip_id", "condition", "seed")):
            res = {"status": "refused", "clip_id": rt["clip_id"], "condition": rt["condition"], "seed": rt["seed"],
                   "note": f"{st}: the output was made for another clip, condition or seed, so it is not shown"}
        out.append({"station": st, "output": res})
    return out


# ---- cards: plain words only -------------------------------------------------------------------------------------------------
def card_text(spec: dict, rt: dict, stations: list, enums: dict) -> str:
    if rt["kind"] == "decline":
        reason = spec["decline_reason"]
        return (f"Declined: {plain_reason(reason)}. Nearest tested option: {nearest_words(reason, enums)}. "
                f"StillHere Pilot plans exploratory studies only from its table of tested options.")
    lines = [f"Plan ({VERSION}): {plain_id(spec['population'])}, {plain_id(spec['language'])}, {plain_id(spec['stimulus'])}; "
             f"conditions {', '.join(plain_id(c) for c in spec['conditions'])}; brain areas "
             f"{', '.join(plain_id(b) for b in spec['brain_areas'])}; analysis: {plain_id(spec['analysis'])}."]
    if rt["kind"] == "evidence":
        lines.append(rt["line"])
        return "\n".join(lines)
    said = {"clip": "no clip named, so the first live clip is used", "condition": "no condition named, so the first one listed after intact is used",
            "seed": "no seed named, so the default seed is used"}
    lines.append(f"Live run: {plain_id(rt['clip_id'])}, intact against {plain_id(rt['condition'])}, seed {rt['seed']}"
                 + (f" ({'; '.join(said[d] for d in rt['defaults'])})" if rt["defaults"] else "") + ".")
    for s in stations:
        o = s["output"]
        if o.get("status") in ("pending", "refused"):
            lines.append(f"{s['station'].capitalize()}: {o['status']}.")
        elif s["station"] == "forecast":
            lines.append(f"Forecast, not checked against scans (exploratory): {o.get('wording', 'see the forecast panel')}")
        else:
            lines.append(f"{s['station'].capitalize()}: {o.get('wording', 'done')}.")
    return "\n".join(lines)


# ---- plan --------------------------------------------------------------------------------------------------------------------
def plan(question, backend: str = "spark-1.3", client: v2.Client | None = None, executors: dict | None = None) -> v2.Card:
    """One json_schema call for the v2.1 StudySpec (v2's single re-prompt, then v2's rule-made decline); no tool call. The card's
    results are the stations our code ran for the route. question is a scanned PublicQuestion or an allowlisted PublicText."""
    client = client or v2.live_client()
    model = v2._model(backend)
    trace, rejected, last = [], [], None
    for _ in range(2):
        enums = load_enums()
        offered = {k: enums[k] for k in ("population", "language", "brain_area", "condition", "stimulus", "analysis", "clip_id")}
        offered["condition_live"] = enums["live_condition"]
        system = [v2.public("planner_v21_system")] + ([v2.public("planner_reprompt")] if rejected else [])
        user = [question, egress.TableEnums(enums=offered)] \
            + ([egress.TableEnums(enums={"rejected": sorted(set(rejected))})] if rejected else [])
        try:
            ans = client(egress.Request(model=model, system=system, user=user, schema_name="study_spec_v21",
                                        json_schema=study_schema(enums)))["content"]
        except (LookupError, ValueError, v2.httpx.HTTPError) as e:
            return v2.Card(text=f"Planner unavailable: the model call failed ({type(e).__name__}). Nothing was planned or declined.",
                           spec={}, declined=True, trace=trace + [f"client_error:{type(e).__name__}"])
        last = ans
        codes = validate_answer(ans, enums)
        if not codes:
            spec = decline_spec(ans["decline_reason"]) if ans["decline_reason"] != "none" else ans
            return _card(spec, enums, executors, trace + ["plan:ok" + (":normalised" if spec != ans else "")])
        trace.append("plan:rejected:" + "|".join(codes))
        rejected = codes
    reason = last.get("decline_reason") if isinstance(last, dict) and last.get("decline_reason") in v2.DECLINE_REASONS \
        and last.get("decline_reason") != "none" else v2.fallback_reason(question)
    card = _card(decline_spec(reason), load_enums(), executors, trace)
    card.text += " This decline came from the planner's fixed rules after the model's answers failed the check twice, not from the model."
    return card


def _card(spec: dict, enums: dict, executors: dict | None, trace: list) -> v2.Card:
    rt = route(spec, enums)
    stations = run_stations(rt, executors)
    card = v2.Card(text=card_text(spec, rt, stations, enums), spec=spec, declined=rt["kind"] == "decline", results=stations,
                   trace=trace + [f"route:{rt['kind']}"])
    card.route = rt
    return card
