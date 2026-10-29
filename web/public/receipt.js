// Fills web/public/receipt.html from /api/receipt/<id>; every value is set as text.
"use strict";
(async () => {
  const id = new URLSearchParams(location.search).get("id") || "";
  const t = document.getElementById("receipt");
  t.textContent = "";
  const r = await fetch("/api/receipt/" + encodeURIComponent(id));
  const d = r.ok ? await r.json() : null;
  const rows = d ? Object.entries(d) : [["receipt", "not found (receipts are kept 14 days)"]];
  const label = {station: "Station", model: "Model", planner: "Planner", spec_sha256: "Input hash (the plan's spec)", input_sha256: "Input hash (built audio)",
                 recipe: "Recipe", seconds: "Time (s)", usd: "API cost (USD)", usd_reserved: "API cost reserved (USD; no published price)",
                 frames: "Frames", gpu: "GPU", utc: "When (UTC)"};
  for (const [k, v] of rows) {
    const tr = document.createElement("tr");
    const a = document.createElement("th"); a.textContent = label[k] || k;
    const b = document.createElement("td"); b.textContent = typeof v === "object" && v !== null ? JSON.stringify(v) : String(v);
    tr.append(a, b); t.append(tr);
  }
})();
