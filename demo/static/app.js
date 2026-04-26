import { setCloud } from "/static/viewer.js";

const form = document.getElementById("chat");
const input = document.getElementById("prompt");
const send = document.getElementById("send");
const meta = document.getElementById("meta");

async function checkHealth() {
  try {
    const r = await fetch("/health");
    const j = await r.json();
    meta.textContent = `ready  |  ${j.device}`;
  } catch (e) {
    meta.textContent = "offline";
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
    meta.textContent = `${data.n_splats} splats  |  ${data.words.join(" ")}`;
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

checkHealth();
// Kick off with a default scene so the viewer isn't empty on load.
window.addEventListener("load", () => {
  input.value = "a red sphere above a blue cube";
  submit(input.value);
});
