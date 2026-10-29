"""The live stations of the hosted Pilot page: PLAN, BUILD, AUDIT, FORECAST, FILM, and the EVIDENCE the static page already shows.
Each live call goes through the typed egress, a spend reservation (web/ledger.py) and the wording guard (web/guard.py); when a call
must not or cannot run (no key, a cap, the session limit, an API error) the station answers with a recorded run, labelled so.
Typed text is used for the one model call it is for and is never stored, logged or put in a receipt."""
import base64
import contextlib
import datetime
import hashlib
import json
import os
import secrets
import struct
import tempfile
import threading
import time
import zlib
from pathlib import Path

import httpx

from stillhere import egress, manipulation_check as MC, meta_client, planner, planner_eval as PE, planner_v21 as P
from web import guard, jobs, ledger
from web.store import get_store

REPO = Path(__file__).resolve().parents[1]
REPLAY = REPO / "web/replay"
PROVIDER = {"spark-1.3": "meta", "spark-open": "fireworks"}
RESERVE = {"plan": 0.05, "audit": 0.03, "transcribe": 0.10, "sam": 0.50}   # worst case $ per call; Voice Transcribe and SAM publish no
RESHUFFLE_SEEDS = (20261028, 7, 1, 2, 3, 4)                                # price here, so they keep their reservation (never settled);
                                                                           # 7 is the third example chip's seed
FILM_FRAMES = REPO / "web/film/frames.json"
RECORDED = "recorded run"
ND = guard.DISCLAIMER
OLD_BUG = {"audio": "assets/conditions/alice_ch01_75s_word_scramble.wav", "results": "results/conditions_manipulation_check.json",
           "commits": ("2002a8d", "64f3c99")}


class Bad(ValueError):
    """A malformed page request (400)."""


# ---- usage capture: the token counts of this thread's model calls, for the spend ledger -------------------------------------------
_tl = threading.local()
_post = httpx.post


def _recording_post(*a, **k):
    r = _post(*a, **k)
    if getattr(_tl, "on", False):
        try:
            u = r.json().get("usage")
        except Exception:
            u = None
        if u:
            _tl.usage.append(u)
    return r


httpx.post = _recording_post


@contextlib.contextmanager
def capture():
    _tl.on, _tl.usage = True, []
    try:
        yield _tl.usage
    finally:
        _tl.on = False


def chat_cost(usage: list) -> float:
    return sum(PE.cost(u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0) for u in usage)


def live_ok(provider: str) -> bool:
    if os.environ.get("STILLHERE_LIVE", "1") == "0":
        return False
    try:
        return bool(meta_client._key()) if provider == "meta" else bool(os.environ.get("FIREWORKS_API_KEY"))
    except Exception:
        return False


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def receipt(rec: dict) -> str:
    rid = secrets.token_hex(8)
    get_store().set(f"receipt:{rid}", json.dumps({**rec, "utc": _now()}), ttl=14 * 86400)
    return rid


def get_receipt(rid: str):
    v = get_store().get(f"receipt:{rid}") if rid and rid.isalnum() else None
    return json.loads(v) if v else None


def _load(rel: str):
    p = REPO / rel
    return json.loads(p.read_text()) if p.is_file() else None


def why_line(code: str) -> str:
    return {"session_limit": "This session has used its 5 live runs for this hour.", "spend_cap": "Today's live budget is used up.",
            "no_key": "Live calls are off on this server.", "api_error": "The live call failed.", "no_run": "Start with a plan first.",
            "station_used": "This run already used this station.", "provider_not_offered": "That model is not offered live.",
            "recorded_plan": "This plan is a recorded run, so this station shows its recording."}.get(
        code, "Live calls are off right now.")


# ---- PLAN -----------------------------------------------------------------------------------------------------------------------
SCOREBOARD = "Planner v2 · 62 unseen requests · 2.3% vs 7.6% / 6.8% (5% bar)"


def menu() -> dict:
    """The live-menu line above the input box, the 3 example chips Fork P ran on both backends, and the scoreboard with its badge."""
    m, ex = _load("stillhere/live_menu.json") or {}, _load("results/planner_v21_examples.json") or {}
    return {"text": P.menu_text(), "chips": [e["text"] for e in ex.get("examples", [])], "badge": P.BADGE, "scoreboard": SCOREBOARD,
            "clips": [{"id": c["id"], "title": c["title"], "seconds": c.get("dur_s")} for c in m.get("clips", [])],
            "default_seed": m.get("default_seed"), "reshuffle_seeds": list(RESHUFFLE_SEEDS)}


def card_view(card) -> dict:
    """What the page shows of a v2.1 card: its own plain-words text (no spec code shown raw), the route, and the guard on every line."""
    rt = getattr(card, "route", None) or {"kind": "unavailable"}
    unavailable = not card.spec
    view = {"decision": "unavailable" if unavailable else "decline" if rt["kind"] == "decline" else "accept",
            "lines": [guard.clean(l) for l in card.text.split("\n") if l.strip()], "route": {"kind": rt["kind"]}}
    if rt["kind"] == "live":
        view["route"].update({k: rt[k] for k in ("clip_id", "condition", "seed", "defaults")})
    elif rt["kind"] == "evidence":                                                   # ruling (2): off the live list -> its evidence panel
        view["route"].update(panel=rt.get("panel"), line=guard.clean(rt.get("line") or ""))
    return view


def recorded_plan(why: str) -> dict:
    rp = _load("web/replay/plan.json") or {}
    return {"mode": "recorded", "label": RECORDED, "why": why_line(why), "runs": rp.get("runs", [])}


def plan(text: str, backend: str, sid: str) -> dict:
    text = (text or "").strip()
    if not text or len(text) > 280:
        raise Bad("text")
    if backend not in PROVIDER:
        raise Bad("backend")
    try:
        q = egress.PublicQuestion(text=text)
    except ValueError:
        return {"mode": "refused", "line": "This request can't be sent to the planner. StillHere plans research studies in healthy "
                                           "volunteers with public stimuli."}
    try:
        if not live_ok(PROVIDER[backend]):                     # checked first, so a recorded run never uses up a visitor's live runs
            raise ledger.Refused("no_key")
        run_id = ledger.start_run(sid); ledger.use(run_id, sid, "plan")
        res = ledger.reserve(PROVIDER[backend], RESERVE["plan"])
    except ledger.Refused as e:
        return recorded_plan(e.code)
    t0 = time.time()
    try:
        with capture() as usage:
            card = P.plan(q, backend, planner.live_client())
    except Exception:
        return recorded_plan("api_error")
    usd = chat_cost(usage); ledger.settle(res, usd)
    if not card.spec:                                          # "planner unavailable": the call failed, so show the recorded runs
        return recorded_plan("api_error")
    view = card_view(card)
    get_store().set(f"runspec:{run_id}", json.dumps({"spec": card.spec or {}, "route": getattr(card, "route", None)}), ttl=ledger.RUN_TTL_S)
    rid = receipt({"station": "plan", "model": P.MODELS[backend], "planner": P.VERSION, "spec_sha256": hashlib.sha256(
        json.dumps(card.spec or {}, sort_keys=True).encode()).hexdigest(), "seconds": round(time.time() - t0, 2), "usd": round(usd, 5)})
    return {"mode": "live", "run_id": run_id, "backend": backend, "card": view, "badge": P.BADGE, "receipt_id": rid, "nd": ND}


# ---- AUDIT ----------------------------------------------------------------------------------------------------------------------
AUDIT_SCHEMA = {"type": "object", "properties": {"notes": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
                "required": ["notes"], "additionalProperties": False}


def audit(run_id: str, sid: str) -> dict:
    """Spark Open's advisory check of the plan: never blocks anything, never measured."""
    label = "Advisory check, not measured"
    spec = (json.loads(get_store().get(f"runspec:{run_id}") or "null") or {}).get("spec") if run_id else None
    if not spec:                                               # a recorded plan (no live run): the recorded audit
        return {**(_load("web/replay/audit.json") or {"notes": []}), "mode": "recorded", "label": f"{label} ({RECORDED})",
                "why": "This plan is a recorded run, so its check is too."}
    if (spec.get("decline_reason") or "none") != "none":
        return {"mode": "none", "label": label, "line": "No accepted plan to check."}
    try:
        if not live_ok("fireworks"):
            raise ledger.Refused("no_key")
        ledger.use(run_id, sid, "audit")
        res = ledger.reserve("fireworks", RESERVE["audit"])
    except ledger.Refused as e:
        return {**(_load("web/replay/audit.json") or {"notes": []}), "mode": "recorded", "label": f"{label} ({RECORDED})", "why": why_line(e.code)}
    enums = {k: [v] if isinstance(v, str) else list(v) for k, v in spec.items()
             if k in ("population", "language", "stimulus", "analysis", "brain_areas", "conditions", "clip_id", "condition") and v}
    model = P.MODELS["spark-open"]
    t0 = time.time()
    try:
        with capture() as usage:
            out = meta_client.chat(egress.Request(model=model, system=[planner.public("audit_system")],
                                                  user=[egress.TableEnums(enums=enums)], schema_name="audit_notes",
                                                  json_schema=AUDIT_SCHEMA))
    except Exception:
        return {**(_load("web/replay/audit.json") or {"notes": []}), "mode": "recorded", "label": f"{label} ({RECORDED})", "why": why_line("api_error")}
    usd = chat_cost(usage); ledger.settle(res, usd)
    notes = [guard.clean(n) for n in (out.get("notes") or [])[:3] if isinstance(n, str)]
    rid = receipt({"station": "audit", "model": model, "seconds": round(time.time() - t0, 2), "usd": round(usd, 5)})
    return {"mode": "live", "label": label, "notes": notes, "receipt_id": rid}


# ---- BUILD ----------------------------------------------------------------------------------------------------------------------
def _recipe_mod():
    import importlib
    return importlib.import_module("worker.recipe")


def checklist(condition: str, chk: dict) -> list:
    """The Voice Transcribe checklist: each item with its number and the rule it is held to (stillhere/manipulation_check.py RULES)."""
    r, pc = chk["rule"], lambda x: f"{round(100 * x)}%"
    if condition == "reversed":
        return [{"item": "Words heard on reversed speech", "value": str(chk["n_words_heard"]), "ok": None, "rule": "no word check"}]
    items = [{"item": "Words heard (word match)", "value": pc(chk["recall"]), "ok": chk["recall"] >= r["recall_min"],
              "rule": f"at least {pc(r['recall_min'])}"}]
    if condition == "intact":
        items.append({"item": "Order kept", "value": pc(chk["order_vs_original"]), "ok": chk["order_vs_original"] >= r["order_vs_original_min"],
                      "rule": f"at least {pc(r['order_vs_original_min'])}"})
    elif condition == "sentence_scramble":
        items += [{"item": "Sentences heard in the new order", "value": pc(chk["order_vs_expected"]),
                   "ok": chk["order_vs_expected"] >= r["order_vs_expected_min"], "rule": f"at least {pc(r['order_vs_expected_min'])}"},
                  {"item": "Order changed from the original", "value": pc(chk["order_vs_expected"] - chk["order_vs_original"]),
                   "ok": chk["order_vs_expected"] - chk["order_vs_original"] >= r["expected_minus_original_min"],
                   "rule": f"new order beats the old by at least {pc(r['expected_minus_original_min'])}"}]
    elif condition == "word_scramble":
        items.append({"item": "Order changed from the original", "value": pc(chk["order_vs_original"]),
                      "ok": chk["order_vs_original"] <= r["order_vs_original_max"], "rule": f"at most {pc(r['order_vs_original_max'])} in the old order"})
    return items


def old_bug() -> dict:
    """The real broken word scramble, from the committed audio and its committed check: no model call, numbers copied."""
    c = (_load(OLD_BUG["results"]) or {})["conditions"]["word_scramble"]
    return {"mode": "recorded", "label": "the old bug, from the repository history", "condition": "word_scramble", "pass": False,
            "checklist": checklist("word_scramble", c), "audio_url": "/media/old_bug.wav",
            "caption": (f"Our first word scramble (commit {OLD_BUG['commits'][0]}, Oct 28) kept only {round(100 * c['recall'])}% of its "
                        f"words when Voice Transcribe read it back (commit {OLD_BUG['commits'][1]}), so it was caught before scanning.")}


def build(run_id: str, sid: str, clip_id: str, condition: str, seed: int) -> dict:
    try:
        r = jobs.recipe(clip_id, condition, seed)
    except ValueError as e:
        raise Bad(str(e))
    if condition in ("sentence_scramble", "word_scramble") and r["seed"] not in RESHUFFLE_SEEDS:
        raise Bad("seed_not_offered")
    rec = _load("web/replay/build.json") or {}
    recorded = lambda why: {**rec.get(f"{clip_id}:{condition}", rec.get("default", {})), "mode": "recorded", "label": RECORDED, "why": why_line(why)}
    try:
        if not live_ok("meta"):
            raise ledger.Refused("no_key")
        ledger.use(run_id, sid, "build", max_uses=1 + len(RESHUFFLE_SEEDS))
    except ledger.Refused as e:
        return recorded("recorded_plan" if e.code == "no_run" else e.code)
    R = _recipe_mod(); idx = R.load_index()
    built, orig = R.build_condition(r, idx), R.build_condition({"clip_id": clip_id, "condition": "intact", "seed": 0}, idx)
    wav = R.wav_bytes(built.audio, built.sr)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"{clip_id}_{condition}.wav"; p.write_bytes(wav)
        try:
            audio = egress.PublicAudio(path=str(p))                    # a build that is not on the allowlist is refused: "build differs"
        except ValueError:
            return {"mode": "refused", "line": "This build differs from the listed public builds, so it was not sent."}
        try:
            ledger.reserve("meta", RESERVE["transcribe"])            # no published price: the reservation stands
        except ledger.Refused as e:
            return recorded(e.code)
        try:
            out = meta_client.transcribe(audio, mode="DIARIZATION")
        except Exception:
            return recorded("api_error")
    chk = MC.check_condition(condition, out.get("transcript") or "", [w["word"] for w in built.words], [w["word"] for w in orig.words])
    items = checklist(condition, chk)
    rid = receipt({"station": "build", "model": meta_client.ASR_MODEL, "input_sha256": hashlib.sha256(wav).hexdigest(),
                   "recipe": r, "seconds": round(time.time() - t0, 2), "usd": None, "usd_reserved": RESERVE["transcribe"]})
    return {"mode": "live", "condition": condition, "seed": r["seed"], "pass": chk["pass"], "checklist": items,
            "audio_b64": base64.b64encode(wav).decode(), "receipt_id": rid}


# ---- FORECAST -------------------------------------------------------------------------------------------------------------------
REVERSED_LINE = "0 words reach the text lane (model input, not a brain result)"
WORD_RATE_LINE = "Flat by construction: a word-rate model cannot see order."
GRAY_LINE = "No public scan of this clip. Forecast, not a scan."
CHIP = {"short": "Forecast only · audio-only, laptop", "long": GRAY_LINE}


def forecast(run_id: str, sid: str, clip_id: str, condition: str, seed: int) -> dict:
    try:
        r = jobs.recipe(clip_id, condition, seed)
    except ValueError as e:
        raise Bad(str(e))
    base = {"chip": CHIP, "end_line": GRAY_LINE, "word_rate_line": WORD_RATE_LINE,
            "reversed_line": REVERSED_LINE if condition == "reversed" else None, "hours_et": hours_et()}
    try:
        ledger.use(run_id, sid, "forecast", max_uses=4)
    except ledger.Refused as e:                                # no live run (a recorded plan) or used up: the cached forecast only
        return {**base, "state": "offline", "why": "Cached forecast only." if e.code == "no_run" else why_line(e.code),
                "fallback": jobs.fallback(r)}
    return {**base, **jobs.submit(r, sid)}


def hours_et() -> str:
    return os.environ.get("WORKER_HOURS_ET", "{worker_hours_et}")


# ---- FILM -----------------------------------------------------------------------------------------------------------------------
def png(rgb) -> bytes:
    """A minimal RGB PNG (standard library zlib), so the server needs no imaging library to send an image."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) \
        + chunk(b"IEND", b"")


CREDIT = "Tears of Steel (CC) Blender Foundation | mango.blender.org, CC BY 3.0. Edited for this study."


def film(run_id: str, sid: str, concept: str) -> dict:
    concept = (concept or "").strip()
    if not concept or len(concept) > 40:
        raise Bad("concept")
    try:
        q = egress.PublicQuestion(text=concept)
    except ValueError:
        return {"mode": "refused", "credit": CREDIT, "line": "This concept can't be sent."}
    rec = _load("web/replay/film.json") or {"frames": []}
    recorded = lambda why: {**rec, "mode": "recorded", "label": RECORDED, "why": why_line(why), "credit": CREDIT}
    try:
        if not live_ok("meta"):
            raise ledger.Refused("no_key")
        if not run_id:
            run_id = ledger.start_run(sid)
        ledger.use(run_id, sid, "film")
        import cv2
        from stillhere import face_edit as FE
        import meta_sam_parser  # noqa: F401  (Meta's SAM License; a dependency, not included)
    except ledger.Refused as e:
        return recorded(e.code)
    except ImportError:
        return recorded("no_key")
    from web import sam
    frames, t0, spent = [], time.time(), 0.0
    for f in json.loads(FILM_FRAMES.read_text())["frames"][:3]:
        try:
            ledger.reserve("meta", RESERVE["sam"]); spent += RESERVE["sam"]        # no published price: the reservation stands
            seg = sam.segment(str(REPO / f["clip"]), q)
        except ledger.Refused as e:
            return recorded(e.code)
        except Exception:
            return recorded("api_error")
        frame = cv2.cvtColor(cv2.imread(str(REPO / f["release_jpg"])), cv2.COLOR_BGR2RGB)
        if not seg["objects"]:
            frames.append({"id": f["id"], "found": False, "line": "no mask found", "intact": _b64png(frame)}); continue
        masks = FE.masks_for({"objects": seg["objects"]})
        img, px, offs = FE.edit(frame, masks, skip_unplaceable=True)
        frames.append({"id": f["id"], "found": True, "n_masks": len(masks), "controls": sum(o is not None for o in offs),
                       **{v: _b64png(img[v]) for v in FE.VERSIONS}})
    rid = receipt({"station": "film", "model": "sam-3.1", "frames": [f["id"] for f in frames], "seconds": round(time.time() - t0, 2),
                   "usd": None, "usd_reserved": round(spent, 2)})
    return {"mode": "live", "frames": frames, "credit": CREDIT, "receipt_id": rid,
            "caption": "SAM 3.1 masked what you typed; StillHere filled it with the surrounding colour, and filled a same-size patch "
                       "elsewhere in the frame the same way as a control."}


def _b64png(rgb) -> str:
    return "data:image/png;base64," + base64.b64encode(png(rgb[::2, ::2].copy())).decode()


# ---- EVIDENCE -------------------------------------------------------------------------------------------------------------------
def evidence() -> dict:
    """The static page's evidence panels (one builder, scripts/build_pilot_page.py), the ghost-line data where it exists, the order-map
    outcome, and which tile sits beside a live gray line: the notthefall pair only if D20 passed, else the LPP ghost line (ruling 3)."""
    bp, sl = guard._bp, {k: _load(v) for k, v in guard._bp.SLOTS.items()}
    html = (bp.lpp_html(_load("results/section8.json"), _load("results/section8_baselines.json"), _load("results/ghostline_lpp_s8.json"))
            + bp.wall_html(sl["story_wall"])
            + bp.d20_html(sl["d20"]) + bp.film_result_html(sl["face_map"], sl["film_score"], _load("results/ghostline_film.json")))
    d20 = bp.d20_status(sl["d20"])
    om = _load("results/order_map_gonogo.json") or {}
    return {"html": html, "lpp_ghost": _load("results/ghostline_lpp_s8.json"), "film_ghost": _load("results/ghostline_film.json"),
            "d20": d20, "pair": "d20" if d20["status"] == "pass" else "lpp",
            "pair_label": "Real listeners, different story." if d20["status"] == "pass" else "Real listeners, held-out story",
            "order_map": {"verdict": om.get("verdict"), "consequence": om.get("consequence")}}
