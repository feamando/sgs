/**
 * Satz demo v0.1 - Frontend logic
 * Handles prompt submission, blob display, and k-slider interaction.
 */

(function () {
  "use strict";

  const promptEl = document.getElementById("prompt");
  const generateBtn = document.getElementById("generate-btn");
  const outputEl = document.getElementById("output");
  const kSlider = document.getElementById("k-slider");
  const kValueEl = document.getElementById("k-value");
  const blobListEl = document.getElementById("blob-list");
  const statusEl = document.getElementById("status");
  const deviceInfoEl = document.getElementById("device-info");
  const tempInput = document.getElementById("temp-input");
  const maxTokensInput = document.getElementById("max-tokens-input");

  // ── Health check on load ──
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      const data = await res.json();
      if (data.ok) {
        deviceInfoEl.textContent =
          (data.model_params || "?") + " params | " +
          (data.n_blobs ? data.n_blobs.toLocaleString() + " blobs" : "?") +
          " | " + (data.device || "?");
        statusEl.textContent = "Ready";
      } else {
        deviceInfoEl.textContent = "server error";
        statusEl.textContent = "Server not ready";
      }
    } catch (e) {
      deviceInfoEl.textContent = "disconnected";
      statusEl.textContent = "Cannot reach server";
    }
  }

  // ── k-slider ──
  kSlider.addEventListener("input", function () {
    kValueEl.textContent = this.value;
  });

  // ── Generate ──
  async function doGenerate() {
    const prompt = promptEl.value.trim();
    if (!prompt) return;

    generateBtn.disabled = true;
    statusEl.textContent = "Generating...";
    outputEl.innerHTML = '<span class="output-empty">Generating, please wait...</span>';
    blobListEl.innerHTML = '<span class="blob-empty">Retrieving blobs...</span>';

    const k = parseInt(kSlider.value, 10);
    const temperature = parseFloat(tempInput.value) || 0.8;
    const maxNew = parseInt(maxTokensInput.value, 10) || 200;

    try {
      const res = await fetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt,
          k: k,
          max_new: maxNew,
          temperature: temperature,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "Request failed");
      }

      const data = await res.json();
      renderOutput(data);
      renderBlobs(data.blobs);
      statusEl.textContent =
        "Done: " + data.generated_tokens + " tokens generated, " +
        data.blobs.length + " blobs retrieved";
    } catch (e) {
      outputEl.innerHTML = '<span class="output-empty">Error: ' + escapeHtml(e.message) + "</span>";
      statusEl.textContent = "Error: " + e.message;
    } finally {
      generateBtn.disabled = false;
    }
  }

  // ── Render output text ──
  function renderOutput(data) {
    const promptSpan = document.createElement("span");
    promptSpan.className = "prompt-echo";
    promptSpan.textContent = data.prompt;

    const genSpan = document.createElement("span");
    genSpan.className = "generated";
    genSpan.textContent = data.generated_text;

    outputEl.innerHTML = "";
    outputEl.appendChild(promptSpan);
    outputEl.appendChild(genSpan);
  }

  // ── Render blob list ──
  function renderBlobs(blobs) {
    if (!blobs || blobs.length === 0) {
      blobListEl.innerHTML = '<span class="blob-empty">No blobs retrieved.</span>';
      return;
    }

    blobListEl.innerHTML = "";
    for (const blob of blobs) {
      const row = document.createElement("div");
      row.className = "blob-row";

      const barWidth = Math.max(2, blob.score_normalized * 100);

      row.innerHTML =
        '<div class="blob-header">' +
        '  <span class="blob-rank">#' + blob.rank + "</span>" +
        '  <span class="blob-score-label">' + blob.score.toFixed(4) + "</span>" +
        "</div>" +
        '<div class="blob-bar-container">' +
        '  <div class="blob-bar" style="width: ' + barWidth + '%"></div>' +
        "</div>" +
        '<div class="blob-meta">' +
        '  <span class="blob-index">idx ' + blob.index + "</span>" +
        '  <span>norm ' + blob.feature_norm.toFixed(2) + "</span>" +
        "</div>";

      blobListEl.appendChild(row);
    }
  }

  // ── Helpers ──
  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // ── Event bindings ──
  generateBtn.addEventListener("click", doGenerate);

  promptEl.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      doGenerate();
    }
  });

  // ── Init ──
  checkHealth();
})();
