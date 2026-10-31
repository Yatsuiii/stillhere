"""Study planner v2 (StillHere Pilot): a researcher types the study they want to run; the planner turns it into a
StudySpec, declines anything outside what has been validated, and runs the stimulus tools.
v2 (docs/rules/planner_v2_eval_rule.md) changes two things from v0: a schema-valid decline is normalised (fields nulled,
the model's reason kept) instead of rejected, and the StudySpec has an analysis field from the validation table.

Two calls at most per question, both through the typed egress wrapper, both served by meta_client:
1. plan: one json_schema call for the StudySpec. The enums are rebuilt from stillhere/validation_table.json on every
   request, so a table edit needs no code change. A rejected answer is re-prompted exactly once; after that a
   deterministic decline card ships (never an unchecked answer). A decline is never rejected for leaving fields filled:
   the planner nulls them and keeps the model's reason.
2. run_tools: when the spec is not a decline, one call offering the four stimulus tools (parallel_tool_calls=False).
   The planner executes the returned calls deterministically and in order; tool results go to the researcher's card
   only, never back to a model.

Planner requests carry PublicText, PublicQuestion and TableEnums only: never TierD (TierD can never go to the
Fireworks or Baseten backends). Reasoning effort is "low" on every backend.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx
import numpy as np

from . import conditions as condlib
from . import egress, meta_client

TABLE = Path(__file__).with_name("validation_table.json")
PT = Path(__file__).with_name("public_text")
RESULTS = Path(__file__).resolve().parent.parent / "results"
ASSETS = Path(__file__).resolve().parent.parent / "assets"
LIBRIVOX_WAV = ASSETS / "librivox_alice_ch01_75s.wav"
LIBRIVOX_WORDS = ASSETS / "librivox_alice_ch01_75s.words.json"

MODELS = {"spark-1.3": "muse-spark-1.3", "spark-open": "fireworks/muse-spark-open", "spark-open-baseten": "baseten/muse-spark-open-1026"}
EFFORT = "low"

DECLINE_REASONS = ("none", "population_not_validated", "language_not_validated", "brain_area_not_validated",
                   "condition_not_validated", "stimulus_not_validated", "analysis_not_offered", "clinical_out_of_scope")
REJECT_CODES = ("schema_invalid", "decline_inconsistent")

Client = Callable[[egress.Request], dict]

_cache: dict = {}


def public(name: str) -> egress.PublicText:
    return egress.PublicText(text=(PT / f"{name}.txt").read_text())


def live_client(timeout: float = 60.0) -> Client:
    return lambda req: meta_client.call(req, timeout=timeout, effort=EFFORT)


# ---- enums ------------------------------------------------------------------
def load_enums(table: Path | None = None) -> dict:
    """Rebuilt from the validation table on every request: population validated only; language validated or
    group_level_only; brain_area, condition and stimulus validated only. failed_check, pending and not_offered ids
    never appear. analysis: validated only (the StudySpec's analysis field)."""
    t = json.loads(Path(table or TABLE).read_text())
    ok = lambda e: e.get("status") == "validated"
    return {"population": [e["id"] for e in t.get("population", []) if ok(e)],
            "language": [e["id"] for e in t.get("language", []) if e.get("status") in ("validated", "group_level_only")],
            "brain_area": [e["id"] for e in t.get("brain_area", []) if ok(e)],
            "condition": [e["id"] for e in t.get("condition", []) if ok(e)],
            "stimulus": [e["id"] for e in t.get("stimulus", []) if ok(e)],
            "analysis": [e["id"] for e in t.get("analysis", []) if ok(e)]}


def _enum(vals) -> dict:
    return {"type": "string", "enum": sorted(set(vals))}


def study_schema(enums: dict) -> dict:
    """ONE flat object: additionalProperties false, every property required. Nullable fields use a type list with
    null and an enum that includes null; array fields may be empty."""
    nullable = lambda vals: {"type": ["string", "null"], "enum": sorted(set(vals)) + [None]}
    return {"type": "object", "additionalProperties": False,
            "required": ["population", "language", "stimulus", "analysis", "brain_areas", "conditions", "decline_reason"],
            "properties": {"population": nullable(enums["population"]),
                           "language": nullable(enums["language"]),
                           "stimulus": nullable(enums["stimulus"]),
                           "analysis": nullable(enums["analysis"]),
                           "brain_areas": {"type": "array", "items": _enum(enums["brain_area"])},
                           "conditions": {"type": "array", "items": _enum(enums["condition"])},
                           "decline_reason": _enum(DECLINE_REASONS)}}


def validate_answer(ans, enums: dict) -> list:
    """Client-side validator: the answer must match the schema ([schema_invalid]); an acceptance must be complete
    ([decline_inconsistent]): population, language, stimulus and analysis set, at least 2 conditions including "intact".
    A schema-valid decline always passes; plan() normalises it (v2)."""
    if not isinstance(ans, dict) or set(ans) != {"population", "language", "stimulus", "analysis", "brain_areas",
                                                 "conditions", "decline_reason"}:
        return ["schema_invalid"]
    if (ans.get("population") not in enums["population"] + [None]
            or ans.get("language") not in enums["language"] + [None]
            or ans.get("stimulus") not in enums["stimulus"] + [None]
            or ans.get("analysis") not in enums["analysis"] + [None]
            or not isinstance(ans.get("brain_areas"), list)
            or not all(isinstance(v, str) and v in enums["brain_area"] for v in ans["brain_areas"])
            or not isinstance(ans.get("conditions"), list)
            or not all(isinstance(v, str) and v in enums["condition"] for v in ans["conditions"])
            or ans.get("decline_reason") not in DECLINE_REASONS):
        return ["schema_invalid"]
    if ans["decline_reason"] != "none":
        return []
    if ans["population"] is not None and ans["language"] is not None and ans["stimulus"] is not None and ans["analysis"] is not None \
            and len(ans["conditions"]) >= 2 and "intact" in ans["conditions"]:
        return []
    return ["decline_inconsistent"]


# ---- deterministic decline ----------------------------------------------------
_TIMING = re.compile(r"\b(lag|lags|delay|delays?|delayed|timing|onset|offset|shift|shifted)\b|[+-]\s*\d", re.I)
_CLINICAL = re.compile(r"\b(patients?|clinic|clinical|diagnos|therap|treatment|treats?|disorder|disease|symptom|doctors?|"
                       r"hospital|stroke|aphasia|dementia)\b", re.I)
_AGE = re.compile(r"year[\s-]?olds?|\b(child|children|kids?|infant|baby|babies|toddler|teens?|adolescent|elderly|"
                  r"ages?|aged)\b", re.I)
_OTHER_LANGUAGE = re.compile(r"\b(german|spanish|italian|mandarin|chinese|cantonese|japanese|hindi|arabic|portuguese|"
                             r"russian|dutch|korean|swedish|norwegian|danish|finnish|polish|turkish|greek|hebrew|"
                             r"bengali|tamil|thai|vietnamese|indonesian|swahili|bilingual|multilingual)\b", re.I)
_OTHER_BRAIN = re.compile(r"\b(visual|motor|frontal|prefrontal|parietal|temporal|occipital|hippocampus|amygdala|"
                          r"cerebellum|thalamus|cortex|broca|wernicke|fusiform|retina)\b", re.I)
_AXIS_REASON = {"population": "population_not_validated", "language": "language_not_validated",
                "brain_area": "brain_area_not_validated", "condition": "condition_not_validated",
                "stimulus": "stimulus_not_validated", "analysis": "analysis_not_offered"}


def fallback_reason(question, table: Path | None = None) -> str:
    """The deterministic decline reason for a question, from the question text and the table: unvalidated ids named in
    the text map by their axis; otherwise timing words decline the analysis, clinical words decline the clinic, age
    words decline the population. Defaults to condition_not_validated."""
    text = getattr(question, "text", "") or ""
    if _CLINICAL.search(text):
        return "clinical_out_of_scope"
    if _TIMING.search(text):
        return "analysis_not_offered"
    if _AGE.search(text):
        return "population_not_validated"
    t = json.loads(Path(table or TABLE).read_text())
    low = text.lower()
    for axis in ("population", "language", "brain_area", "condition", "stimulus", "analysis"):
        allowed = {"validated"} if axis != "language" else {"validated", "group_level_only"}
        for e in t.get(axis, []):
            if e.get("status") not in allowed and e["id"].lower() in low:
                return _AXIS_REASON[axis]
    if _OTHER_LANGUAGE.search(text):
        return "language_not_validated"
    if _OTHER_BRAIN.search(text):
        return "brain_area_not_validated"
    return "condition_not_validated"


def decline_spec(reason: str) -> dict:
    return {"population": None, "language": None, "stimulus": None, "analysis": None,
            "brain_areas": [], "conditions": [], "decline_reason": reason}


def nearest_option(reason: str, enums: dict) -> str:
    axis = {"population_not_validated": "population", "language_not_validated": "language",
            "brain_area_not_validated": "brain_area", "condition_not_validated": "condition",
            "stimulus_not_validated": "stimulus", "analysis_not_offered": "analysis",
            "clinical_out_of_scope": "population"}.get(reason, "stimulus")
    opts = enums.get(axis, [])
    return f"{axis} {opts[0]}" if opts else f"{axis} (no validated option)"


# ---- cards --------------------------------------------------------------------
@dataclass
class Card:
    text: str
    spec: dict
    declined: bool                                   # no plan to run tools on: a decline, or the planner was unavailable
    results: list = field(default_factory=list)
    trace: list = field(default_factory=list)

    def display(self) -> str:
        return self.text


def build_card(spec: dict, results: list, enums: dict) -> Card:
    declined = spec.get("decline_reason", "none") != "none"
    if declined:
        reason = spec["decline_reason"]
        text = (f"Declined: {reason}. Nearest validated option: {nearest_option(reason, enums)}. "
                f"StillHere Pilot plans exploratory studies only from the validation table.")
        return Card(text=text, spec=spec, declined=True, results=[], trace=["decline:" + reason])
    lines = [f"StudySpec: population={spec['population']}, language={spec['language']}, "
             f"brain_areas={spec['brain_areas']}, conditions={spec['conditions']}, stimulus={spec['stimulus']}, "
             f"analysis={spec.get('analysis')}."]
    for r in results:
        out = r["output"]
        if r["tool"] == "list_stimuli":
            lines.append(f"Stimuli (exploratory): {', '.join(out['stimuli'])}.")
        elif r["tool"] == "make_conditions":
            built = ", ".join(f"{k} ({v['samples']} samples)" for k, v in out["conditions"].items())
            lines.append(f"Conditions from {out['source']} (exploratory): {built}.")
        elif r["tool"] == "check_conditions":
            lines.append(f"Manipulation check (exploratory): {json.dumps(out.get('conditions', out), sort_keys=True)}.")
        elif r["tool"] == "forecast":
            lines.append(f"Forecast, not validated (exploratory): {out.get('wording', json.dumps(out, sort_keys=True))}")
    return Card(text="\n".join(lines), spec=spec, declined=False, results=results, trace=["plan:ok"])


# ---- plan ---------------------------------------------------------------------
def _model(backend: str) -> str:
    if backend not in MODELS:
        raise ValueError(f"backend is {' or '.join(sorted(MODELS))}, not {backend!r}")
    return MODELS[backend]


def plan(question, backend: str = "spark-1.3", client: Client | None = None, system: str = "planner_system") -> Card:
    """One json_schema call for the StudySpec; a rejected answer is re-prompted exactly once, then a deterministic
    decline card ships. A schema-valid decline is normalised, not rejected. question is an allowlisted menu PublicText or a
    scanned PublicQuestion. system names the system prompt: v2's by default; R4's candidate is planner_system_r4."""
    client = client or live_client()
    model = _model(backend)
    trace, rejected, last = [], [], None
    for _ in range(2):
        enums = load_enums()
        schema = study_schema(enums)
        offered = {"population": enums["population"], "language": enums["language"],
                   "brain_area": enums["brain_area"], "condition": enums["condition"], "stimulus": enums["stimulus"],
                   "analysis": enums["analysis"]}
        prompt = [public(system)] + ([public("planner_reprompt")] if rejected else [])
        user = [question, egress.TableEnums(enums=offered)] \
            + ([egress.TableEnums(enums={"rejected": sorted(set(rejected))})] if rejected else [])
        try:
            ans = client(egress.Request(model=model, system=prompt, user=user,
                                        schema_name="study_spec", json_schema=schema))["content"]
        except (LookupError, ValueError, httpx.HTTPError) as e:     # a failed call is not a decline: never guess a reason
            return Card(text=f"Planner unavailable: the model call failed ({type(e).__name__}). Nothing was planned or declined.",
                        spec={}, declined=True, trace=trace + [f"client_error:{type(e).__name__}"])
        last = ans
        codes = validate_answer(ans, enums)
        if not codes and ans["decline_reason"] != "none":
            spec = decline_spec(ans["decline_reason"])              # v2: the model's decline, fields nulled
            card = build_card(spec, [], enums)
            if spec != ans:
                card.trace = ["plan:ok:normalised"] + card.trace
            return card
        trace.append("plan:ok" if not codes else "plan:rejected:" + "|".join(codes))
        if not codes:
            return build_card(ans, [], enums)
        rejected = codes
    reason = last.get("decline_reason") if isinstance(last, dict) and last.get("decline_reason") in DECLINE_REASONS \
        and last.get("decline_reason") != "none" else fallback_reason(question)
    enums = load_enums()
    card = build_card(decline_spec(reason), [], enums)
    card.text += " This decline came from the planner's fixed rules after the model's answers failed the check twice, not from the model."
    card.trace = trace + card.trace
    return card


# ---- tools --------------------------------------------------------------------
def _cached_json(name: str) -> dict:
    if name not in _cache:
        _cache[name] = json.loads((RESULTS / name).read_text())
    return _cache[name]


def _as_list(v) -> list:
    return v if isinstance(v, list) else [v]


def _tool_list_stimuli(args: dict, spec: dict, enums: dict) -> dict:
    return {"stimuli": list(enums["stimulus"]), "label": "exploratory"}


def _tool_make_conditions(args: dict, spec: dict, enums: dict) -> dict:
    import hashlib

    want = [c for a in _as_list(args.get("condition", spec.get("conditions", []))) for c in _as_list(a)]
    want = [c for c in want if c in enums["condition"]] or [c for c in spec.get("conditions", [])]
    x, sr = condlib.read_wav(LIBRIVOX_WAV)
    words = json.loads(LIBRIVOX_WORDS.read_text())["words"]
    built = condlib.build(x, sr, words)
    out = {}
    for name in want:
        b = np.ascontiguousarray(built[name])
        y = np.clip(np.round(b * 32768.0), -32768, 32767).astype("<i2")
        out[name] = {"samples": int(len(b)), "sha256": hashlib.sha256(y.tobytes()).hexdigest()}
    return {"conditions": out, "source": LIBRIVOX_WAV.name, "label": "exploratory"}


def _tool_check_conditions(args: dict, spec: dict, enums: dict) -> dict:
    return _cached_json("conditions_manipulation_check.json")


def _tool_forecast(args: dict, spec: dict, enums: dict) -> dict:
    return _cached_json("live_tribe_librivox.json")


_EXEC = {"list_stimuli": _tool_list_stimuli, "make_conditions": _tool_make_conditions,
         "check_conditions": _tool_check_conditions, "forecast": _tool_forecast}


def tool_defs(enums: dict) -> list:
    return [egress.Tool(name="list_stimuli", description=public("tool_list_stimuli"), params={"stimulus": enums["stimulus"]}),
            egress.Tool(name="make_conditions", description=public("tool_make_conditions"),
                        params={"condition": enums["condition"]}),
            egress.Tool(name="check_conditions", description=public("tool_check_conditions"),
                        params={"condition": enums["condition"]}),
            egress.Tool(name="forecast", description=public("tool_forecast"), params={"stimulus": enums["stimulus"]})]


def run_tools(spec: dict, backend: str = "spark-1.3", client: Client | None = None) -> list:
    """When the spec is not a decline, one call offering the four tools; the returned calls run deterministically and
    in order, and their results go to the researcher's card only, never back to a model."""
    if (spec.get("decline_reason") or "none") != "none":
        return []
    client = client or live_client()
    model = _model(backend)
    enums = load_enums()
    user = [egress.TableEnums(enums={k: [v] if isinstance(v, str) else list(v)
                                           for k, v in {"population": spec["population"], "language": spec["language"],
                                                        "stimulus": spec["stimulus"], "analysis": spec.get("analysis") or [],
                                                        "brain_areas": spec["brain_areas"],
                                                        "conditions": spec["conditions"],
                                                        "decline_reason": spec["decline_reason"]}.items()})]
    req = egress.Request(model=model, system=[public("planner_system")], user=user,
                         tools=tool_defs(enums), parallel_tool_calls=False)
    try:
        calls = client(req).get("tool_calls", [])
    except (LookupError, ValueError, httpx.HTTPError):
        return []
    out = []
    for c in calls:
        name = (c.get("name") or "")
        if name in _EXEC:
            try:
                out.append({"tool": name, "output": _EXEC[name](c.get("arguments") or {}, spec, enums)})
            except (LookupError, ValueError, OSError):
                continue
    return out
