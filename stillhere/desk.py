"""Muse Spark analysis desk (SPARK-PLAN A; PREREG §6, §7, D11, D18, D21 Part B, D23). Spark decides what may be said; it never writes it.

Two calls per question, both through the typed egress wrapper:
1. Route: the question and the analysis catalogue go out with two tools. Spark calls run_registered once per analysis it needs
   (parallel calls), or decline_unregistered with the nearest registered analysis. tool_choice is auto-only on this API, so the desk
   forces tool use: a reply without a usable call is re-prompted once.
2. Answer: the looked-up frozen values go out as tier-D, with a strict schema rebuilt from the frozen bar results for this request. Its
   line_id enum holds only licensed lines of the analyses looked up (a passed bar's line, a failed bar's frozen fallback, the §7 sedation
   line only if D11 licenses it, and nothing but P7's fallback if P7 failed), so an unlicensed line cannot be decoded. When the answer
   shows the no-stage-line outcome, the desk adds any failed co-primary bar's frozen claim Spark left out, and orders the lines.
Spark never types a digit: each number slot is bound to a tool output key, the desk checks every binding against the line's own, and the
renderer fills the number from the frozen results. A rejected answer is re-prompted once; after that the deterministic template report
ships through the same verifier. Spark's note is scanned and never shown. Lookups only: nothing is computed live.
Stage default is REPLAY (recorded answers looked up by request digest) with a visible badge; LIVE is optional."""
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, NamedTuple

import httpx

from . import egress, meta_client, verifier
from .frozen import ANALYSES, Frozen

PT = Path(__file__).with_name("public_text")
MODEL = "muse-spark-1.3"
DECLINE_REASONS = ("none", "not_preregistered", "clinical_out_of_scope")
STAGE_BARS = ("sedation", "P1", "P3", "P8", "P2a")          # what the §7 sedation line rests on (D11), for declines


def public(name: str) -> egress.PublicText:
    return egress.PublicText(text=(PT / f"{name}.txt").read_text())


# ---- lines --------------------------------------------------------------------------------------------------------------
class Line(NamedTuple):
    analysis: str
    text: str                  # template; every number is a {slot}
    slots: dict                # slot -> (field of the analysis' lookup, format)


_FMT = {"i": lambda v: str(int(v)), "p": lambda v: f"{v:.2g}", "z": lambda v: f"{v:+.2f}", "f2": lambda v: f"{v:.2f}", "min": lambda v: f"{v:.1f}"}
_TIME = {"trs": ("trs", "i"), "minutes": ("minutes", "min")}
_SED = "With zero control scans, StillHere saw language fade under deep sedation in {k} of {n} volunteers while hearing held"
LINES = {
    "sedation_stage": Line("sedation", _SED + " ({trs} TRs = {minutes} minutes per state)", {"k": ("k", "i"), "n": ("n", "i"), **_TIME}),
    "sedation_stage_waking": Line("sedation", _SED + " and come back on waking ({trs} TRs = {minutes} minutes per state)",
                                  {"k": ("k", "i"), "n": ("n", "i"), **_TIME}),
    "sedation_no_line": Line("sedation", "Gate 1: no sedation stage line; the frozen If-it-fails claims apply (Deep effective N {deep_n}; "
                             "{trs} TRs = {minutes} minutes per state)", {"deep_n": ("deep_effective_n", "i"), **_TIME}),
}
_P, _N, _THR = ("p", "p"), ("n", "i"), ("holm_threshold", "p")
# bar -> (label, the §6 "If it fails" claim, detail with its number slots); P4's claim is in plain words (cloud ruling of Oct 29, item 3)
BARS = {
    "P7": ("P7 (pipeline sanity, read first)", "Fix preprocessing before reading anything else",
           "FPN LOO-ISC Awake > Deep p = {p_fpn}, n = {n_fpn}; Deep AUD LOO-ISC above its null p = {p_aud}, n = {n_aud} (brain-to-brain ISC, not TRIBE)",
           {"p_fpn": ("p_fpn_awake_gt_deep", "p"), "n_fpn": ("n_fpn", "i"), "p_aud": ("p_aud_deep_gt_0", "p"), "n_aud": ("n_aud_deep", "i")}),
    "P1": ("P1 (co-primary)", 'HEARING + ISC replication; "TRIBE-only LANGUAGE did not reach the bar"',
           "LANGUAGE Awake > Deep in {k} of {n} volunteers (needed {k_star}), p = {p}, Hodges-Lehmann median change {hl} z [95% CI {hl_lo}, {hl_hi}]",
           {"k": ("k", "i"), "n": _N, "k_star": ("k_star", "i"), "p": _P, "hl": ("hl", "z"), "hl_lo": ("hl_ci_lo", "z"), "hl_hi": ("hl_ci_hi", "z")}),
    "P3": ("P3 (co-primary)", "One combined gauge, no split",
           "LANGUAGE minus HEARING change from Awake to Deep, Hodges-Lehmann median {hl} z [95% CI {lo}, {hi}], n = {n}",
           {"hl": ("hl", "z"), "lo": ("ci_lo", "z"), "hi": ("ci_hi", "z"), "n": _N}),
    "P8": ("P8 (co-primary)", "Name the gauge SPEECH-LOCKED; do not claim it follows the word sequence",
           "Awake LANGUAGE with the real words > shuffled words in {k} of {n} volunteers, p = {p}, Hodges-Lehmann {hl} z [95% CI {hl_lo}, {hl_hi}]",
           {"k": ("k", "i"), "n": _N, "p": _P, "hl": ("hl", "z"), "hl_lo": ("hl_ci_lo", "z"), "hl_hi": ("hl_ci_hi", "z")}),
    "P2a": ("P2a", "At most one global stimulus-tracking gauge", "TRIBE HEARING in Deep above the rest null, p = {p} (Holm threshold {thr}), n = {n}",
            {"p": _P, "thr": _THR, "n": _N}),
    "P2b": ("P2b", "No individual-level claim", "TRIBE's individual detection rate against LOO-ISC's on the same runs, both gauges, n = {n}",
            {"n": ("H.n", "i")}),
    "P4": ("P4", '"group-level result only; not claimed for one person"',
           "LANGUAGE AUC Awake vs Deep {auc} [95% CI {lo}, {hi}], delta AUC vs LOO-ISC {dauc}, n = {n}",
           {"auc": ("L.auc", "f2"), "lo": ("L.auc_ci_lo", "f2"), "hi": ("L.auc_ci_hi", "f2"), "dauc": ("L.delta_auc", "z"), "n": ("L.n", "i")}),
    "P5_H": ("P5 HEARING", "Limitation; downgrade the affected gauge", "N3 ratio {r3}, N4 ratio {r4}, p = {p} (Holm threshold {thr}), n = {n}",
             {"r3": ("n3.ratio", "f2"), "r4": ("n4.ratio", "f2"), "p": _P, "thr": _THR, "n": ("n3.n", "i")}),
    "P5_L": ("P5 LANGUAGE", "Limitation; downgrade the affected gauge", "N3 ratio {r3}, N4 ratio {r4}, p = {p} (Holm threshold {thr}), n = {n}",
             {"r3": ("n3.ratio", "f2"), "r4": ("n4.ratio", "f2"), "p": _P, "thr": _THR, "n": ("n3.n", "i")}),
    "P6": ("P6", "Awake vs Deep only; state the repetition confound",
           "LANGUAGE Recovery > Deep p = {p} (Holm threshold {thr}), median Recovery {mr} z, median Awake {ma} z, n = {n}",
           {"p": _P, "thr": _THR, "mr": ("median_recovery", "z"), "ma": ("median_awake", "z"), "n": _N}),
}
for _aid, (_label, _claim, _detail, _slots) in BARS.items():
    _met = "bar met under Holm" if ANALYSES[_aid].holm else "bar met"
    LINES[f"{_aid}_pass"] = Line(_aid, f"{_label}: {_met}; " + _detail + " ({trs} TRs = {minutes} minutes per state)", {**_slots, **_TIME})
    LINES[f"{_aid}_fallback"] = Line(_aid, f"{_label}: bar not met, so the claim becomes: {_claim} (" + _detail
                                     + "; {trs} TRs = {minutes} minutes per state)", {**_slots, **_TIME})


def licensed_lines(fz: Frozen) -> list[str]:
    """Each bar's line if it passed, else its frozen fallback; the sedation line only if D11 licenses it; only P7's fallback if P7 failed."""
    if not fz.passed("P7"):
        return ["P7_fallback"]                                # D11: P7 fails -> read nothing else
    g = fz.g
    sed = ("sedation_stage_waking" if g["come_back_on_waking"] else "sedation_stage") if g["sedation_stage_line"] else "sedation_no_line"
    return [sed] + [f"{aid}_{'pass' if fz.passed(aid) else 'fallback'}" for aid in BARS]


def template_lines(fz: Frozen) -> list[str]:
    """The deterministic report: the D11 outcome, with the fallback claims of the failed co-primary bars and P2a."""
    lic = licensed_lines(fz)
    if lic == ["P7_fallback"] or lic[0].startswith("sedation_stage"):
        return lic[:1]
    return [lic[0]] + [f"{aid}_fallback" for aid in ("P1", "P3", "P8", "P2a") if f"{aid}_fallback" in lic]


def bindings(lid: str) -> dict:
    line = LINES[lid]
    return {slot: f"{line.analysis}.{fld}" for slot, (fld, _) in line.slots.items()}


LAG_MARK = " Lag-sensitive (D24): its pass or fail differs at L +/- 1 s or at dL = 0."
TIMING_WORDS = re.compile(r"\b(lag|lags|delay|delays|delayed|timing|offset|shift|shifted|onset|seconds?|s)\b|[+-]\s*\d", re.I)
LAG_DECLINE = "P1 carries the pre-registered lag-sensitive marks at L - 1 s and L + 1 s (D24); they are reported beside its result."


def timing_question(q) -> bool:
    """A question about the lag or timing. Its decline always points to P1 (the D24 lag marks at L - 1 s and L + 1 s), whatever
    analysis Spark names: the desk decides this, not the model."""
    return bool(TIMING_WORDS.search(getattr(q, "text", "")))
TIMING_NOTE = "Gate 1 timing: L set by D24; scored before the 9 PM ET decision-table label, recorded in D25."


def render(fz: Frozen, lid: str) -> str:
    """Fill a line's numbers from the frozen values its bindings name; P8 failing renames the gauge in every line (stricter reading).
    A lag-sensitive bar carries the D24 mark beside its result; the desk adds it, Spark never chooses it."""
    vals = fz.lookup(LINES[lid].analysis)
    nums = {slot: "n/a" if vals.get(key) is None else _FMT[LINES[lid].slots[slot][1]](vals[key]) for slot, key in bindings(lid).items()}
    text = LINES[lid].text.format(**nums) + (LAG_MARK if fz.lag_sensitive(LINES[lid].analysis) else "")
    if not fz.passed("P8"):
        text = text.replace("language readout", "speech-locked readout").replace("LANGUAGE", "SPEECH-LOCKED")
    return text


def _enum(vals) -> dict:
    return {"type": "string", "enum": sorted(set(vals))}


def answer_schema(lids: list[str], sources: list[str]) -> dict:
    """Strict: additionalProperties false and every property required, at every level."""
    binding = {"type": "object", "additionalProperties": False, "required": ["line_id", "slot", "source"],
               "properties": {"line_id": _enum(lids), "slot": _enum(s for lid in lids for s in LINES[lid].slots), "source": _enum(sources)}}
    return {"type": "object", "additionalProperties": False, "required": ["decline_reason", "nearest_analysis", "line_ids", "bindings", "note"],
            "properties": {"decline_reason": _enum(DECLINE_REASONS), "nearest_analysis": _enum(ANALYSES), "line_ids": {"type": "array", "items": _enum(lids)},
                           "bindings": {"type": "array", "items": binding}, "note": {"type": "string"}}}


TOOLS = [egress.Tool(name="run_registered", description=public("tool_run_registered"), params={"analysis_id": list(ANALYSES)}),
         egress.Tool(name="decline_unregistered", description=public("tool_decline_unregistered"), params={"nearest_analysis": list(ANALYSES)})]


# ---- clients ------------------------------------------------------------------------------------------------------------
Client = Callable[[egress.Request], dict]


class NotRecorded(LookupError):
    pass


ADDED_DEFAULTS = {"parallel_tool_calls": True}          # Request fields added after sessions were recorded, with their defaults


def digest(req: egress.Request) -> str:
    """The replay key of a request: SHA-256 of its canonical JSON. A field added later is left out while it holds its default, so
    sessions recorded before it existed keep their keys."""
    d = req.model_dump(mode="json")
    for k, v in ADDED_DEFAULTS.items():
        if d.get(k) == v:
            d.pop(k)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


class ReplayClient:
    """Answers from a JSONL of {"digest", "response"} recorded earlier (the recorder is a later step); unknown requests raise NotRecorded."""
    def __init__(self, path: Path):
        lines = Path(path).read_text().splitlines() if Path(path).exists() else []
        self.rec = {r["digest"]: r["response"] for r in map(json.loads, filter(str.strip, lines))}

    def __call__(self, req: egress.Request) -> dict:
        if digest(req) not in self.rec:
            raise NotRecorded(digest(req)[:12])
        return self.rec[digest(req)]


class RecordingClient:
    """Wraps a live client and appends {"digest", "response"} for every answered request, so a live session becomes a REPLAY file."""
    def __init__(self, inner: Client, path: Path):
        self.inner, self.path = inner, Path(path)

    def __call__(self, req: egress.Request) -> dict:
        out = self.inner(req)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({"digest": digest(req), "response": out}) + "\n")
        return out


def live_client(timeout: float = 20.0, effort: str | None = None) -> Client:
    return lambda req: meta_client.call(req, timeout=timeout, effort=effort)


# ---- desk ---------------------------------------------------------------------------------------------------------------
@dataclass
class Card:
    text: str
    controls_scanned: int
    subject: dict
    badge: str
    source: str                                               # "spark", "decline" or "template"
    trace: list = field(default_factory=list)                 # codes only, no content

    def verifier_view(self) -> dict:
        return {"text": self.text, "subject": self.subject, "controls_scanned": self.controls_scanned}

    def display(self) -> str:
        return f"[{self.badge}]\n{self.text}"


class Desk:
    def __init__(self, fz: Frozen, client: Client | None = None, mode: str = "REPLAY", replay: Path | None = None, model: str = MODEL):
        if mode not in ("REPLAY", "LIVE"):
            raise ValueError("mode is REPLAY or LIVE")
        self.fz, self.mode, self.model = fz, mode, model
        self.client = client or (live_client() if mode == "LIVE" else ReplayClient(replay or Path("replay/desk.jsonl")))

    # cards
    def _card(self, body: list[str], controls: int, source: str, trace: list) -> Card:
        badge = " · ".join([self.mode] + (["SYNTHETIC DATA"] if self.fz.synthetic else []) + (["TEMPLATE"] if source == "template" else []))
        body = body + ([TIMING_NOTE] if self.fz.timing_notes else [])
        return Card(text=verifier.HEADER.format(n=controls) + "\n" + "\n".join(body), controls_scanned=controls,
                    subject={"kind": "group", "n": self.fz.included_awake}, badge=badge, source=source, trace=trace)

    def _lines_card(self, lids: list[str], source: str, trace: list) -> Card:
        return self._card([render(self.fz, lid) for lid in lids], max(self.fz.controls(LINES[lid].analysis) for lid in lids), source, trace)

    def template(self, trace: list) -> Card:
        card = self._lines_card(template_lines(self.fz), "template", trace)
        bad = verifier.verify(card.verifier_view())
        if bad:
            raise AssertionError(f"template report fails the verifier: {bad}")     # a bug in the fixed lines, never shown
        return card

    def decline(self, aid: str, trace: list) -> Card:
        timing = getattr(self, "_timing", False)        # set per question in ask()
        if timing:
            aid = "P1"
        body = [f"Not pre-registered, so StillHere does not run it. Nearest registered analysis: {aid} (PREREG {ANALYSES[aid].prereg})."]
        stage = [lid for lid in licensed_lines(self.fz) if lid.startswith("sedation_stage")]
        body.append(render(self.fz, stage[0]) if aid in STAGE_BARS and stage else LAG_DECLINE if timing else "Its result is reported in the README.")
        return self._card(body, 0, "decline", trace)

    # steps
    def _route(self, q, trace: list):
        system, user = [public("desk_route"), public("desk_catalogue")], [q]
        for _ in range(2):
            calls = self.client(egress.Request(model=self.model, system=system, user=user, tools=TOOLS)).get("tool_calls", [])
            aids = [c["arguments"].get("analysis_id") for c in calls if c.get("name") == "run_registered"]
            aids = [a for a in aids if a in ANALYSES]
            dec = [c["arguments"].get("nearest_analysis") for c in calls if c.get("name") == "decline_unregistered"]
            dec = next((a for a in dec if a in ANALYSES), None)
            trace.append(f"route:{len(calls)}_calls:{len(aids)}_lookups" + (":decline" if dec else ""))
            if aids or dec:
                return [a for a in ANALYSES if a in aids], dec
            system, user = system + [public("desk_reprompt")], [q, egress.TierD(enums={"rejected.0": "no_tool_call"})]
        return None, None

    def _check(self, ans: dict, lids: list[str], values: dict, trace: list):
        """Returns (card, []) or (None, violation codes)."""
        v = []
        note = ans.get("note", "")
        if re.search(r"\d", note):
            v.append("digit_in_note")
        if verifier.blocked_words(note):
            v.append("blocked_word_in_note")
        reason, near = ans.get("decline_reason"), ans.get("nearest_analysis")
        if reason not in DECLINE_REASONS or near not in ANALYSES:
            v.append("bad_decline_fields")
        if v:
            return None, v
        if reason == "clinical_out_of_scope":
            return self._card([verifier.DECLINE_CLINICAL], 0, "decline", trace), []
        if reason == "not_preregistered":
            return self.decline(near, trace), []
        chosen = ans.get("line_ids") or []
        v += ["no_line"] if not chosen else []
        v += ["unlicensed_line" for lid in chosen if lid not in lids]
        given = {}
        for b in ans.get("bindings", []):
            if b.get("source") not in values:
                v.append("unknown_source")                    # a number that no tool returned
            elif b.get("line_id") not in chosen:
                v.append("stray_binding")
            else:
                given.setdefault(b["line_id"], {})[b.get("slot")] = b["source"]
        for lid in (lid for lid in chosen if lid in lids):
            want, got = bindings(lid), given.get(lid, {})
            v += [f"missing_binding:{lid}.{s}" for s in want if s not in got]
            v += [f"wrong_binding:{lid}.{s}" for s in want if s in got and got[s] != want[s]]
            v += ["stray_binding" for s in got if s not in want]
        if v:
            return None, v
        order = licensed_lines(self.fz)
        if "sedation_no_line" in chosen:                      # the outcome always comes with the failed co-primary bars' frozen claims
            added = [lid for lid in template_lines(self.fz) if lid not in chosen]
            trace += [f"desk_added:{lid}" for lid in added]
            chosen = chosen + added
        card = self._lines_card(sorted(chosen, key=order.index), "spark", trace)
        return (card, []) if not verifier.verify(card.verifier_view()) else (None, ["verifier_rejected"])

    def _answer(self, q, aids: list[str], trace: list) -> Card:
        values = {k: x for a in aids for k, x in self.fz.lookup(a).items()}
        lids = [lid for lid in licensed_lines(self.fz) if LINES[lid].analysis in aids]
        guide = {f"line.{lid}.{s}": key for lid in lids for s, key in bindings(lid).items()}
        schema = answer_schema(lids, list(values))
        data, rejected = egress.TierD(values=values, enums=guide), []
        for _ in range(2):
            system = [public("desk_answer"), public("desk_catalogue")] + ([public("desk_reprompt")] if rejected else [])
            user = [q, data] + ([egress.TierD(enums={f"rejected.{i}": c for i, c in enumerate(rejected)})] if rejected else [])
            ans = self.client(egress.Request(model=self.model, system=system, user=user, schema_name="desk_answer", json_schema=schema))["content"]
            card, rejected = self._check(ans, lids, values, trace)
            trace.append("answer:ok" if card else "answer:rejected:" + "|".join(sorted(set(c.split(":")[0] for c in rejected))))
            if card:
                return card
        return self.template(trace)

    def ask(self, q) -> Card:
        """q: an allowlisted menu question (egress.PublicText) or a typed, scanned egress.PublicQuestion."""
        trace = []; self._timing = timing_question(q)
        try:
            aids, dec = self._route(q, trace)
            if aids is None:
                return self.template(trace)
            if not aids:
                return self.decline(dec, trace)
            if not self.fz.passed("P7"):
                aids = ["P7"]                                 # D11: read nothing else
            return self._answer(q, aids, trace)
        except (LookupError, ValueError, httpx.HTTPError) as e:   # not recorded, refused by egress, malformed reply, network
            trace.append(f"client_error:{type(e).__name__}")
            return self.template(trace)
