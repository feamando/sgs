import { setCloud, setMarkers, setMarkersVisible } from "/static/viewer.js";

const form = document.getElementById("chat");
const input = document.getElementById("prompt");
const send = document.getElementById("send");
const meta = document.getElementById("meta");
const objectsPanel = document.getElementById("objects");
const warningsPanel = document.getElementById("warnings");
const markersToggle = document.getElementById("markers-toggle");
const dslEditor = document.getElementById("dsl-editor");
const dslErrors = document.getElementById("dsl-errors");
const dslRenderBtn = document.getElementById("dsl-render");

// Tab switching
document.querySelectorAll(".side-tabs .tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".side-tabs .tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(tab.dataset.tab).classList.add("active");
  });
});

async function checkHealth() {
  try {
    const r = await fetch("/health");
    const j = await r.json();
    meta.textContent = `ready | ${j.device}`;
  } catch (e) {
    meta.textContent = "offline";
  }
}

function renderObjectsPanel(data) {
  if (!objectsPanel) return;
  objectsPanel.innerHTML = "";
  if (!data.objects || data.objects.length === 0) {
    objectsPanel.innerHTML = '<div class="obj-empty">no objects recognised</div>';
    return;
  }
  for (const o of data.objects) {
    const row = document.createElement("div");
    row.className = "obj-row";
    const sw = document.createElement("span");
    sw.className = "obj-swatch";
    const r = Math.round(o.color[0] * 255);
    const g = Math.round(o.color[1] * 255);
    const b = Math.round(o.color[2] * 255);
    sw.style.background = `rgb(${r},${g},${b})`;
    const label = document.createElement("span");
    label.className = "obj-label";
    const pos = o.position.map((x) => x.toFixed(2)).join(", ");
    label.innerHTML =
      `<b>${o.template}</b> <span class="obj-word">&ldquo;${o.word}&rdquo;</span>` +
      `<span class="obj-meta">(${pos}) conf ${Math.round(o.confidence * 100)}%</span>`;
    row.appendChild(sw);
    row.appendChild(label);
    objectsPanel.appendChild(row);
  }
}

function renderWarnings(data) {
  if (!warningsPanel) return;
  warningsPanel.innerHTML = "";
  if (!data.warnings || data.warnings.length === 0) return;
  for (const w of data.warnings) {
    const row = document.createElement("div");
    row.className = "warning-row";
    row.textContent = w;
    warningsPanel.appendChild(row);
  }
}

async function submit(prompt) {
  send.disabled = true;
  meta.textContent = "rendering...";
  try {
    const r = await fetch("/generate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      meta.textContent = `error: ${err.detail || r.statusText}`;
      return;
    }
    const data = await r.json();
    setCloud(data.splats);
    setMarkers(data.objects);
    renderObjectsPanel(data);
    renderWarnings(data);
    if (data.dsl && dslEditor) {
      dslEditor.value = JSON.stringify(data.dsl, null, 2);
      dslErrors.textContent = "";
    }
    const unresolved = data.n_unresolved ? ` | ${data.n_unresolved} unresolved` : "";
    meta.textContent =
      `${data.n_objects} objects | ${data.n_splats} splats${unresolved} | ` +
      data.words.join(" ");
  } catch (e) {
    meta.textContent = `error: ${e.message}`;
  } finally {
    send.disabled = false;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const p = input.value.trim();
  if (!p) return;
  submit(p);
});

if (markersToggle) {
  markersToggle.addEventListener("change", () => {
    setMarkersVisible(markersToggle.checked);
  });
}

// DSL edit + re-render
if (dslRenderBtn) {
  dslRenderBtn.addEventListener("click", async () => {
    if (!dslEditor) return;
    let dsl;
    try {
      dsl = JSON.parse(dslEditor.value);
    } catch (e) {
      dslErrors.textContent = `JSON error: ${e.message}`;
      return;
    }
    dslErrors.textContent = "";
    dslRenderBtn.disabled = true;
    meta.textContent = "re-rendering from DSL...";
    try {
      const r = await fetch("/render-dsl", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ dsl }),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({ detail: r.statusText }));
        const detail = err.detail;
        if (typeof detail === "object" && detail.errors) {
          dslErrors.textContent = detail.errors.join("; ");
        } else {
          dslErrors.textContent = detail || r.statusText;
        }
        meta.textContent = "DSL validation failed";
        return;
      }
      const data = await r.json();
      setCloud(data.splats);
      meta.textContent = `${data.n_objects} objects | ${data.n_splats} splats (from DSL edit)`;
    } catch (e) {
      dslErrors.textContent = e.message;
      meta.textContent = "error";
    } finally {
      dslRenderBtn.disabled = false;
    }
  });
}

checkHealth();
window.addEventListener("load", () => {
  input.value = "a red sphere above a blue cube";
  submit(input.value);
});
