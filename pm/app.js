/*
 * Radiance Labs roadmap visualizer.
 *
 * Reads roadmap.md (one level up), parses two pipe-tables:
 *   - "Swimlanes" — lane metadata
 *   - "Entries"   — one row per model/product version
 * and renders two streams (Models, Products) each with a set of lanes.
 *
 * Run via any static file server from repo root so the fetch of
 * ../roadmap.md works (see pm/README.md).
 */

const MODEL_LABEL_ORDER = [1, 2, 3, 5, 6, 7];    // Planck, Hertz, Helmholtz, Klang, Raum, Einstein
const PRODUCT_LABEL_ORDER = [8, 9, 10, 11];      // Prisma, Klang, Raum, Satz

const meta = document.getElementById("meta");
const modelsLanesEl = document.getElementById("models-lanes");
const productsLanesEl = document.getElementById("products-lanes");

init();

async function init() {
  try {
    const res = await fetch("../roadmap.md", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    const { swimlanes, entries } = parseRoadmap(text);
    render(swimlanes, entries);
    meta.textContent =
      `${entries.length} entries | ${swimlanes.length} lanes`;
  } catch (err) {
    meta.textContent = "error";
    modelsLanesEl.innerHTML = "";
    productsLanesEl.innerHTML = "";
    const box = document.createElement("div");
    box.className = "error";
    box.innerHTML =
      `<strong>Could not load <code>../roadmap.md</code>.</strong><br>` +
      `${String(err).replace(/</g, "&lt;")}<br>` +
      `Run a local server from the repo root:<br>` +
      `<code>cd /path/to/sgs && python -m http.server 8000</code><br>` +
      `Then open <code>http://localhost:8000/pm/</code>.`;
    document.querySelector("main").prepend(box);
  }
}

// ── Parser ─────────────────────────────────────────────────────────
/**
 * Find pipe-tables by heading. Only fenced by an H2 directly before
 * the table; body is a contiguous sequence of lines starting with "|".
 */
function parseRoadmap(md) {
  const lines = md.split(/\r?\n/);
  const tables = {};

  let currentHeader = null;
  let collecting = null;  // { rows: [] }

  for (const raw of lines) {
    const line = raw.trimEnd();
    const h = line.match(/^##\s+(.+?)\s*$/);
    if (h) {
      if (collecting) tables[currentHeader] = collecting.rows;
      collecting = null;
      currentHeader = h[1];
      continue;
    }
    if (!currentHeader) continue;
    if (line.startsWith("|")) {
      if (!collecting) collecting = { rows: [] };
      collecting.rows.push(line);
    } else if (collecting && line.trim() === "") {
      // blank — still inside the table group, keep going
    } else if (collecting) {
      // Non-table content ends the table.
      tables[currentHeader] = collecting.rows;
      collecting = null;
    }
  }
  if (collecting) tables[currentHeader] = collecting.rows;

  const swimlanes = parseTable(tables["Swimlanes"] || [], [
    "swimlane-id", "name", "kind", "description",
  ]);
  const entries = parseTable(tables["Entries"] || [], [
    "id", "name", "type", "status", "date_created", "notes",
  ]);

  // Normalise.
  for (const s of swimlanes) {
    s.id = Number(s["swimlane-id"]);
    s.kind = (s.kind || "").toLowerCase();
  }
  for (const e of entries) {
    e.laneId = Number((e.id || "").split("-")[0]);
    e.type = (e.type || "").toLowerCase();
    e.status = normaliseStatus(e.status);
  }
  return { swimlanes, entries };
}

function parseTable(rows, expectedCols) {
  if (rows.length < 2) return [];
  const cells = (line) =>
    line
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((c) => c.trim());
  const headerCells = cells(rows[0]).map((c) => c.toLowerCase());
  // Skip the separator row (|---|---|...).
  const out = [];
  for (let i = 2; i < rows.length; i++) {
    const parts = cells(rows[i]);
    if (parts.length !== headerCells.length) continue;
    const obj = {};
    for (let k = 0; k < headerCells.length; k++) {
      obj[headerCells[k]] = parts[k];
    }
    out.push(obj);
  }
  return out;
}

function normaliseStatus(s) {
  const v = (s || "").toLowerCase().trim();
  if (v === "done" || v === "shipped" || v === "complete" || v === "completed") return "done";
  if (v === "in progress" || v === "wip" || v === "active") return "in progress";
  return "open";
}

// ── Renderer ───────────────────────────────────────────────────────
function render(swimlanes, entries) {
  // Group entries by laneId.
  const byLane = new Map();
  for (const e of entries) {
    if (!byLane.has(e.laneId)) byLane.set(e.laneId, []);
    byLane.get(e.laneId).push(e);
  }
  // Sort each lane by version ascending (by name), open/in-progress after done.
  for (const arr of byLane.values()) arr.sort(cmpEntries);

  const modelLanes = swimlanes.filter((s) => s.kind === "model");
  const productLanes = swimlanes.filter((s) => s.kind === "product");

  modelsLanesEl.innerHTML = "";
  for (const lane of orderLanes(modelLanes, MODEL_LABEL_ORDER)) {
    modelsLanesEl.appendChild(renderLane(lane, byLane.get(lane.id) || []));
  }

  productsLanesEl.innerHTML = "";
  for (const lane of orderLanes(productLanes, PRODUCT_LABEL_ORDER)) {
    productsLanesEl.appendChild(renderLane(lane, byLane.get(lane.id) || []));
  }
}

function orderLanes(lanes, preferredOrder) {
  const byId = new Map(lanes.map((l) => [l.id, l]));
  const out = [];
  for (const id of preferredOrder) {
    if (byId.has(id)) {
      out.push(byId.get(id));
      byId.delete(id);
    }
  }
  // Anything else (unexpected ids) appended at the end by id.
  const rest = [...byId.values()].sort((a, b) => a.id - b.id);
  return out.concat(rest);
}

function renderLane(lane, entries) {
  const el = document.createElement("div");
  el.className = "lane";

  const label = document.createElement("div");
  label.className = "lane-label";
  label.innerHTML =
    `<div class="lane-id">LANE ${String(lane.id).padStart(2, "0")}</div>` +
    `<div class="lane-name">${escapeHtml(lane.name)}</div>` +
    `<div class="lane-count">${entries.length} ${entries.length === 1 ? "entry" : "entries"}</div>`;
  el.appendChild(label);

  const cards = document.createElement("div");
  cards.className = "lane-cards";
  if (entries.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-hint";
    empty.textContent = "(no entries yet)";
    cards.appendChild(empty);
  } else {
    for (const e of entries) cards.appendChild(renderCard(e));
  }
  el.appendChild(cards);
  return el;
}

function renderCard(e) {
  const el = document.createElement("div");
  const statusClass = `status-${e.status.replace(/\s+/g, "-")}`;
  el.className = `card ${statusClass}`;

  const version = extractVersion(e.name);

  el.innerHTML =
    `<div class="card-head">` +
      `<div class="card-name">${escapeHtml(e.name)}</div>` +
      (version ? `<div class="card-version">v${escapeHtml(version)}</div>` : "") +
    `</div>` +
    `<div class="card-pill ${statusClass}">${escapeHtml(e.status)}</div>` +
    `<div class="card-date">${escapeHtml(e.date_created || "")}</div>` +
    (e.notes ? `<div class="card-notes">${escapeHtml(e.notes)}</div>` : "");
  return el;
}

function extractVersion(name) {
  const m = (name || "").match(/([0-9]+(?:\.[0-9]+){0,2})\s*$/);
  return m ? m[1] : "";
}

// ── Helpers ────────────────────────────────────────────────────────
const STATUS_ORDER = { "done": 0, "in progress": 1, "open": 2 };
function cmpEntries(a, b) {
  const va = extractVersion(a.name);
  const vb = extractVersion(b.name);
  const vc = cmpVersion(va, vb);
  if (vc !== 0) return vc;
  return (STATUS_ORDER[a.status] ?? 9) - (STATUS_ORDER[b.status] ?? 9);
}

function cmpVersion(a, b) {
  const pa = (a || "0").split(".").map((n) => parseInt(n, 10) || 0);
  const pb = (b || "0").split(".").map((n) => parseInt(n, 10) || 0);
  const len = Math.max(pa.length, pb.length);
  for (let i = 0; i < len; i++) {
    const d = (pa[i] || 0) - (pb[i] || 0);
    if (d !== 0) return d;
  }
  return 0;
}

function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
