// StillHere Pilot, the live stations rail. Talks only to this page's own server (/api/...); no key is ever in the page.
// Every string from the server is set with textContent, except the evidence panels, which the server renders from committed files.
"use strict";
const $ = (id) => document.getElementById(id);
const S = { run: null, route: null, seeds: [], seed: null, built: false, menu: null, receipts: [] };

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  let data = null;
  try { data = await r.json(); } catch (e) { data = null; }
  return {status: r.status, data: data || {}};
}
function badge(kind, text) { return el("span", "badge b-" + kind, text); }
function nd() { return el("p", "nd", "Not a diagnosis or medical advice."); }
function receiptLink(id) {
  if (!id) return el("span");
  const a = el("a", "small", "receipt");
  a.href = "/receipt.html?id=" + encodeURIComponent(id);
  return a;
}
function recordedNote(d) {
  const p = el("p", "small recorded");
  p.append(badge("wait", d.label || "recorded run"), document.createTextNode(d.why || ""));
  return p;
}
function done(station) { document.querySelector('.rail a[href="#' + station + '"]').classList.add("done"); }

// ---- status and menu ---------------------------------------------------------------------------------------------------------
async function loadStatus() {
  const {data} = await api("/api/status");
  S.menu = data.menu || {};
  $("menu-text").textContent = S.menu.text || "";
  const chips = $("chips"); chips.textContent = "";
  for (const c of S.menu.chips || []) {
    const b = el("button", "", c); b.type = "button";
    b.addEventListener("click", () => { $("ask").value = c; });
    chips.append(b);
  }
  $("scoreboard").textContent = (S.menu.scoreboard || "") + (S.menu.badge ? ". " + S.menu.badge + "." : "");
  const live = data.live || {};
  $("status-line").textContent = "Worker: " + (data.worker_online ? "online" : "offline (cached forecasts)") + ". Live models: " +
    (live.meta ? "on" : "off (recorded runs)") + ". Up to " + data.runs_per_hour + " live runs per hour per visitor.";
  $("hours").textContent = data.saved_line || "";
}

// ---- PLAN ----------------------------------------------------------------------------------------------------------------------
function planCard(d, title) {
  const box = el("div", "panel");
  const c = d.card || {};
  const kind = c.decision === "accept" ? "pass" : c.decision === "decline" ? "info" : "wait";
  box.append(el("h3", "", ""));
  box.firstChild.append(badge(kind, c.decision === "accept" ? "plans it" : c.decision === "decline" ? "declines, with its reason" : "no answer"),
                        document.createTextNode(title || ""));
  box.append(el("div", "card", (c.lines || []).join("\n")));
  return box;
}
async function doPlan() {
  const text = $("ask").value.trim();
  if (!text) return;
  $("plan-go").disabled = true;
  const out = $("plan-out"); out.textContent = "Planning.";
  const {status, data} = await api("/api/plan", {text: text, backend: $("backend").value});
  $("plan-go").disabled = false; out.textContent = "";
  if (status !== 200) { out.append(el("p", "small", "The request could not be planned (" + (data.error || status) + ").")); return; }
  if (data.mode === "refused") { out.append(el("p", "", data.line)); return; }
  if (data.mode === "recorded") {
    out.append(recordedNote(data));
    for (const r of data.runs || []) {
      out.append(el("p", "small", "Recorded request: " + r.request));
      out.append(planCard(r, " " + (r.backend_label || "")));
    }
    const live = (data.runs || []).find((r) => r.card.route && r.card.route.kind === "live" && r.card.route.clip_id);
    if (live) {                                               // carry on down the rail with the first recorded plan
      S.run = null; S.route = live.card.route; S.seed = S.route.seed;
      $("build-recipe").textContent = "Recorded plan: clip " + S.route.clip_id + ", intact against " + S.route.condition.replace("_", " ") + ", seed " + S.route.seed + ".";
      $("build-go").disabled = false; $("audit-go").disabled = false;
    }
    return;
  }
  S.run = data.run_id; S.route = data.card.route; S.receipts.push(data.receipt_id);
  const box = planCard(data, " live");
  box.append(el("p", "small", data.badge || ""), receiptLink(data.receipt_id), nd());
  out.append(box); done("plan");
  if (S.route.kind === "live") {
    S.seed = S.route.seed;
    $("build-recipe").textContent = "Clip " + S.route.clip_id + ", intact against " + S.route.condition.replace("_", " ") + ", seed " + S.route.seed + ".";
    $("build-go").disabled = false; $("audit-go").disabled = false;
  } else if (S.route.kind === "evidence") {
    const p = el("p", "big", S.route.line || "This plan ends on its evidence panel.");
    out.append(p); location.hash = "#evidence";
  }
}

// ---- BUILD ---------------------------------------------------------------------------------------------------------------------
function checklistView(d, cls) {
  const box = el("div", "panel " + (cls || ""));
  const h = el("h3");
  h.append(badge(d.pass === false ? "fail" : d.pass === true ? "pass" : "info", d.pass === false ? "blocked: check failed" : d.pass === true ? "check passed" : "no word check"),
           document.createTextNode(" " + (d.condition || "").replace("_", " ") + (d.seed !== undefined ? ", seed " + d.seed : "")));
  box.append(h);
  const ul = el("ul", "check");
  for (const it of d.checklist || []) {
    const li = el("li", it.ok === true ? "ok" : it.ok === false ? "no" : "", it.item + ": " + it.value + " (rule: " + it.rule + ")");
    ul.append(li);
  }
  box.append(ul);
  if (d.caption) box.append(el("p", "small", d.caption));
  if (d.line) box.append(el("p", "", d.line));
  const src = d.audio_b64 ? "data:audio/wav;base64," + d.audio_b64 : d.audio_url;
  if (src) { const a = el("audio"); a.controls = true; a.src = src; box.append(a); }
  box.append(receiptLink(d.receipt_id));
  return box;
}
async function doBuild(seed) {
  if (!S.route || S.route.kind !== "live") return;
  $("build-go").disabled = true;
  const out = $("build-out"); out.textContent = "Building and transcribing.";
  const {status, data} = await api("/api/build", {run_id: S.run, clip_id: S.route.clip_id, condition: S.route.condition, seed: seed});
  out.textContent = ""; $("build-go").disabled = false;
  if (status !== 200) { out.append(el("p", "small", "Build refused (" + (data.error || status) + ").")); return; }
  if (data.mode === "recorded") out.append(recordedNote(data));
  if (data.mode === "refused") { out.append(el("p", "", data.line)); return; }
  if (data.mode === "live") S.seed = data.seed;               // a recorded build never sets the seed
  S.lastBuild = data; S.built = data.mode === "live" ? data.pass !== false : data.pass === true;
  out.append(checklistView(data));
  $("forecast-go").disabled = !S.built;
  $("reshuffle").disabled = !["sentence_scramble", "word_scramble"].includes(S.route.condition);
  if (S.built) done("build");
}
function nextSeed() {
  const seeds = (S.menu && S.menu.reshuffle_seeds) || [];
  const i = seeds.indexOf(S.seed);
  return seeds[(i + 1) % seeds.length];
}
async function doOldBug() {
  const out = $("build-out");
  const {data} = await api("/api/build", {old_bug: true});
  const view = checklistView(data, "caught flash");
  out.prepend(view);
  setTimeout(() => {
    view.remove();
    if (S.lastBuild) { out.textContent = ""; out.append(checklistView(S.lastBuild)); }
  }, 3000);
}

// ---- AUDIT ---------------------------------------------------------------------------------------------------------------------
async function doAudit() {
  $("audit-go").disabled = true;
  const out = $("audit-out"); out.textContent = "Asking.";
  const {data} = await api("/api/audit", {run_id: S.run});
  out.textContent = "";
  const box = el("div", "panel");
  box.append(el("h3", "", data.label || "Advisory check, not measured"));
  if (data.mode === "recorded") box.append(recordedNote(data));
  if (data.line) box.append(el("p", "", data.line));
  const ul = el("ul");
  for (const n of data.notes || []) ul.append(el("li", "", n));
  box.append(ul, receiptLink(data.receipt_id));
  out.append(box); done("audit");
}

// ---- FORECAST ------------------------------------------------------------------------------------------------------------------
function setLanes(events) {
  for (const lane of document.querySelectorAll(".lane")) {
    const name = lane.dataset.lane;
    const mine = events.filter((e) => e.lane === name);
    const last = mine[mine.length - 1];
    if (!last) continue;
    lane.className = "lane " + last.status;
    lane.querySelector("b").textContent = last.status === "start" ? "running" : last.status === "end" ? "done" : "off";
    if (name === "llama-3.2") {
      const w = mine.map((e) => (e.detail || {}).n_words).filter((x) => x !== undefined);
      if (w.length) lane.querySelector("span").textContent = "Llama 3.2: words in context (" + w[w.length - 1] + " words)";
    }
  }
}
function grayLine(values, chip) {
  const W = 640, H = 220, l = 30, r = 12, t = 34, b = 26;
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "TRIBE forecast, no scan");
  const lo = Math.min(...values), hi = Math.max(...values), pad = (hi - lo) * 0.08 || 1;
  const X = (i) => l + (W - l - r) * i / Math.max(values.length - 1, 1);
  const Y = (v) => t + (H - t - b) * (hi + pad - v) / (hi - lo + 2 * pad);
  const line = document.createElementNS(ns, "polyline");                       // the gray dashed forecast first, alone
  line.setAttribute("points", values.map((v, i) => X(i).toFixed(1) + "," + Y(v).toFixed(1)).join(" "));
  line.setAttribute("fill", "none"); line.setAttribute("stroke", "#8a929b"); line.setAttribute("stroke-width", "2.4"); line.setAttribute("stroke-dasharray", "7 5");
  svg.append(line);
  const c = document.createElementNS(ns, "text");                               // then the chip
  c.setAttribute("x", l); c.setAttribute("y", 18); c.setAttribute("font-size", "14"); c.setAttribute("font-weight", "600");
  c.textContent = chip.short; svg.append(c);
  const x = document.createElementNS(ns, "text");
  x.setAttribute("x", l); x.setAttribute("y", H - 6); x.setAttribute("font-size", "12"); x.setAttribute("fill", "#4a5560");
  x.textContent = "seconds of clip (language areas, TRIBE units)"; svg.append(x);
  return svg;
}
async function showResult(res, d, label) {
  const out = $("forecast-out"); out.textContent = "";
  const end = el("div", "endscreen");
  const mine = el("div", "panel");
  mine.append(el("h3", "", label));
  if (res && res.note) mine.append(el("p", "big", res.note));                // a fallback at another seed says so
  if (res && res.regions) {
    const wrap = el("div", "tbl"); wrap.append(grayLine(res.regions.language, d.chip)); mine.append(wrap);
  }
  mine.append(el("p", "big", d.end_line));
  mine.append(el("p", "small", d.chip.short + ": " + d.chip.long));
  if (d.reversed_line) mine.append(el("p", "small", d.reversed_line));
  mine.append(el("p", "small", d.word_rate_line));
  if (res && res.timings_s) mine.append(el("p", "small", "Computed " + (res.computed_utc || "") + " on " + (res.gpu || "the laptop") + "."));
  mine.append(nd());
  end.append(mine);
  const ev = await api("/api/evidence");
  const pair = el("div", "panel");
  pair.append(el("h3", "", ev.data.pair_label || ""));
  if (ev.data.pair === "d20" && ev.data.d20 && ev.data.d20.line) {
    pair.append(el("p", "", ev.data.d20.line.replace(/`/g, "")));
    if (ev.data.d20.small) pair.append(el("p", "small", ev.data.d20.small));
    if (ev.data.d20.baseline) pair.append(el("p", "small", ev.data.d20.baseline));
    const pre = el("p", "small"); const a = el("a", "", "PREREG.md"); a.href = ev.data.prereg_url;
    pre.append(document.createTextNode(ev.data.d20_chip + ". "), a, document.createTextNode(": " + ev.data.prereg_note)); pair.append(pre);
  } else {
    pair.append(el("p", "small", "Le Petit Prince, section 8, held out: see the evidence panel below."));
  }
  pair.append(nd());
  end.append(pair); out.append(end); done("forecast");
}
async function doForecast() {
  $("forecast-go").disabled = true;
  const out = $("forecast-out"); out.textContent = "Queued.";
  const r = S.route;
  const {status, data} = await api("/api/forecast", {run_id: S.run, clip_id: r.clip_id, condition: r.condition, seed: S.seed});
  if (status >= 400) { out.textContent = "Forecast refused (" + (data.error || status) + ")."; return; }
  $("lane-note").textContent = data.reversed_line || "";
  if (data.state === "cached") return showResult(data.result, data, data.result.source === "cache" ? "Cached forecast, computed " + (data.result.computed_utc || "") : "Forecast");
  if (data.state === "offline" || data.state === "busy") {
    if (data.fallback) await showResult(data.fallback, data, "Cached forecast, computed " + (data.fallback.computed_utc || ""));
    else out.textContent = data.why || "The worker is offline and no cached forecast exists for this clip.";
    return;
  }
  const t0 = Date.now();
  const poll = async () => {
    const s = await api("/api/forecast/" + data.job_id);
    setLanes(s.data.events || []);
    if (s.data.status === "done") return showResult(s.data.result, data, "Forecast, live");
    if (["failed", "expired"].includes(s.data.status)) {
      if (s.data.fallback) return showResult(s.data.fallback, data, "Cached forecast, computed " + (s.data.fallback.computed_utc || ""));
      out.textContent = "The live forecast did not finish."; return;
    }
    out.textContent = (s.data.status === "running" ? "Running on the laptop" : "Queued") + ", " + Math.round((Date.now() - t0) / 1000) + " s.";
    setTimeout(poll, 1500);
  };
  poll();
}

// ---- EVIDENCE and FILM ---------------------------------------------------------------------------------------------------------
async function loadEvidence() {
  const {data} = await api("/api/evidence");
  const box = $("evidence-out");
  box.innerHTML = data.html || "";                                              // rendered by the server from committed results files
  if (data.order_map && data.order_map.verdict) box.append(el("p", "small", "Order map: " + data.order_map.verdict + ". " + (data.order_map.consequence || "")));
}
async function doFilm() {
  const concept = $("concept").value.trim();
  if (!concept) return;
  $("film-go").disabled = true;
  const out = $("film-out"); out.textContent = "Masking three frames.";
  const {status, data} = await api("/api/film", {run_id: S.run, concept: concept});
  $("film-go").disabled = false; out.textContent = "";
  if (status !== 200) { out.append(el("p", "small", "Refused (" + (data.error || status) + ").")); return; }
  const box = el("div", "panel");
  if (data.mode === "recorded") box.append(recordedNote(data));
  if (data.line) box.append(el("p", "", data.line));
  if (data.caption) box.append(el("p", "big", data.caption));
  for (const f of data.frames || []) {
    const row = el("div", "frames frames3");
    const names = [["intact", "Intact"], ["faces_removed", "Filled"], ["area_matched", "Same-size control"]];
    if (!f.found) names.splice(1);
    for (const [k, lab] of names) {
      if (!f[k]) continue;
      const fig = el("figure"); const img = el("img"); img.src = f[k]; img.alt = lab + " frame";
      fig.append(img, el("figcaption", "", lab)); row.append(fig);
    }
    box.append(row);
    if (!f.found) box.append(el("p", "small", "no mask found"));
  }
  box.append(el("p", "credit", data.credit || ""), receiptLink(data.receipt_id), nd());
  out.append(box); done("film");
}

$("plan-go").addEventListener("click", doPlan);
$("build-go").addEventListener("click", () => doBuild(S.seed));
$("reshuffle").addEventListener("click", () => doBuild(nextSeed()));
$("old-bug").addEventListener("click", doOldBug);
$("audit-go").addEventListener("click", doAudit);
$("forecast-go").addEventListener("click", doForecast);
$("film-go").addEventListener("click", doFilm);
loadStatus(); loadEvidence();
