# B1: Tiny Generative SGS Model — Plan (v3)

## Concept

Train a small language model that generates text using SGS rendering, then deploy as a **browser-based app** that runs entirely on the user's device. No server, no API key, no cloud.

**The hook:** *"A language model that renders text like 3D scenes — running in your browser. Just 3D rendering math generating words on your laptop."*

---

## Two Phases

```
Phase A (Build + Train):  Python/PyTorch, train on RTX 4090
Phase B (Deploy):         Export to ONNX, run in browser via WebGPU/ONNX Runtime
```

---

## Why Edge/Browser Works

| Property | SGS LM | GPT-2 Small | Llama 3 8B |
|---|---|---|---|
| Parameters | ~20M | 117M | 8B |
| Size on disk | ~80MB | ~500MB | ~16GB |
| RAM needed | ~200MB | ~1GB | ~32GB |
| CPU inference | ~50ms/token | ~200ms/token | Impossible |
| Runs in browser | **Yes** | Barely | No |

20M params at float16 = ~40MB. With quantization (int8) = ~20MB. A user downloads 20MB once, then generates text locally forever. That's smaller than most website hero images.

---

## Architecture

```
SGSLanguageModel(
  vocab: 30K BPE tokens as Gaussians
  d_s: 64 (splatting space)
  d_f: 256 (feature space)
  n_passes: 2
  context: 256 tokens
)

Generation step for position t:
  1. Activate Gaussians for tokens 1..t-1
  2. Positional modulation per token
  3. Multi-pass rendering (2 passes, causal masking)
  4. Query at position t → rendered meaning vector (d_f=256)
  5. logits = meaning @ vocab_features.T → [30K]
  6. next_token = sample(softmax(logits / temperature))
```

---

## Phase A: Build + Train (7 days + 1 day GPU)

### A.1 Data Pipeline (Day 1)

**TinyStories** (Eldan & Li, 2023): ~2.1M short children's stories. Simple vocabulary, proven at small scale.

```python
# src/tinystories.py
1. Download TinyStories from HuggingFace
2. Train BPE tokenizer (30K vocab) via sentencepiece
3. Tokenize → binary format for fast loading
4. DataLoader: random 256-token crops, causal
```

### A.2 Causal SGS Model (Days 2-3)

```python
# src/sgs_lm.py
class SGSLanguageModel(nn.Module):
    def __init__(self, vocab_size=30000, d_s=64, d_f=256, n_passes=2, max_len=256):
        # Gaussian vocabulary (learned from scratch)
        self.mu = nn.Embedding(vocab_size, d_s)
        self.log_var = nn.Embedding(vocab_size, d_s)
        self.raw_alpha = nn.Embedding(vocab_size, 1)
        self.features = nn.Embedding(vocab_size, d_f)

        # Position embeddings
        self.pos_mu = nn.Embedding(max_len, d_s)

        # Per-pass updates (causal)
        self.mu_update = ...
        self.alpha_gate = ...
        self.ffn = ...

        # Output
        self.log_tau = nn.Parameter(...)
        self.out_proj = nn.Linear(d_f, vocab_size)

    def forward(self, token_ids):
        # For each position t, render meaning from tokens 0..t-1 (causal)
        # Vectorized via causal mask (like transformer causal attention)
        ...
```

**Key optimization: Cached transmittance.** T_t = T_{t-1} × (1 - a_{t-1}). Don't recompute from scratch each position.

### A.3 Training Loop (Day 4)

```python
# scripts/train_lm.py
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

for batch in dataloader:
    logits = model(batch)           # [batch, seq-1, vocab]
    targets = batch[:, 1:]          # shift right
    loss = cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
```

### A.4 Generation (Day 5)

```python
# src/generate.py
def generate(model, prompt_ids, max_new=200, temperature=0.8, top_k=50):
    for _ in range(max_new):
        logits = model(prompt_ids)[:, -1, :]
        logits = logits / temperature
        top_k_vals, top_k_idx = logits.topk(top_k)
        probs = softmax(top_k_vals)
        next_token = top_k_idx[torch.multinomial(probs, 1)]
        prompt_ids = cat([prompt_ids, next_token], dim=1)
    return prompt_ids
```

### A.5 Train (Day 6 — GPU)

```bash
python scripts/train_lm.py \
  --data data/tinystories \
  --d_s 64 --d_f 256 --n_passes 2 \
  --batch_size 32 --lr 3e-4 --epochs 3
```

12-24 hours on RTX 4090. ~$10 electricity.

### A.6 Evaluate + Compare (Day 7)

- Perplexity on validation set
- Generate 100 stories, manual inspection
- Train matched-param transformer LM as baseline
- Question: is SGS text coherent? Is it qualitatively different?

---

## Phase B: Browser Deployment (5-7 days)

### B.1 Export to ONNX (Day 1)

```python
# scripts/export_onnx.py
import torch.onnx

# Export the model with a fixed input shape
dummy_input = torch.randint(0, 30000, (1, 256))
torch.onnx.export(
    model, dummy_input,
    "sgs_lm.onnx",
    input_names=["token_ids"],
    output_names=["logits"],
    dynamic_axes={"token_ids": {1: "seq_len"}, "logits": {1: "seq_len"}},
    opset_version=17,
)
```

Then quantize:
```bash
python -m onnxruntime.quantization.quantize \
  --input sgs_lm.onnx \
  --output sgs_lm_int8.onnx \
  --quantize_mode dynamic
```

Result: ~20MB int8 ONNX model.

### B.2 Browser Runtime (Days 2-3)

**Option A: ONNX Runtime Web (most compatible)**
```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
```
- Works in all browsers (Chrome, Firefox, Safari)
- Uses WebAssembly (WASM) backend for CPU
- WebGPU backend available in Chrome for GPU acceleration

**Option B: Transformers.js (by HuggingFace)**
- Higher-level API, handles tokenization
- May need custom model class for SGS (not a standard architecture)

**Recommendation: ONNX Runtime Web.** More control, no dependency on HuggingFace architecture registry.

```javascript
// inference.js
const session = await ort.InferenceSession.create('sgs_lm_int8.onnx');

async function generateToken(tokenIds) {
    const input = new ort.Tensor('int64', BigInt64Array.from(tokenIds.map(BigInt)), [1, tokenIds.length]);
    const results = await session.run({ token_ids: input });
    const logits = results.logits.data;  // last position
    return sampleTopK(logits, temperature, topK);
}
```

### B.3 Tokenizer in JS (Day 3)

BPE tokenizer needs to run in the browser too.

```javascript
// tokenizer.js
// Load sentencepiece model as a JSON vocab + merge rules
// Implement BPE encode/decode in ~100 lines of JS
// Or use existing: https://github.com/anthropics/anthropic-tokenizer-typescript
```

### B.4 Chat UI (Days 4-5)

```
┌─────────────────────────────────────────┐
│  🔮 SGS Chat                            │
│  "Language generation via 3D rendering"  │
│                                          │
│  ┌─────────────────────────────────────┐ │
│  │ Once upon a time, there was a small │ │
│  │ rabbit who loved to explore. One    │ │
│  │ day, she found a shiny stone near   │ │
│  │ the river...                        │ │
│  │ █ (generating...)                   │ │
│  └─────────────────────────────────────┘ │
│                                          │
│  [Temperature: 0.8] [Top-K: 50]         │
│                                          │
│  Type a prompt: [________________] [Go]  │
│                                          │
│  ⚡ Running locally in your browser      │
│  📊 Model: 20M params, 20MB download     │
│  🧊 No server, no API key, no cloud     │
│                                          │
│  [About] [GitHub] [Report Issue]        │
└─────────────────────────────────────────┘
```

Features:
- Streaming generation (token by token, visible typing effect)
- Temperature / top-k sliders
- Prompt presets ("Tell me a story about...", "Once upon a time...")
- Token/second counter (shows local inference speed)
- "How it works" panel linking to A3 visualizer

### B.5 Merge with A3 Visualizer (Day 6, optional)

If A3 is built first: embed the Gaussian visualization IN the chat app. As the model generates each token, show the Gaussian scene updating in real-time. The user watches the rendering equation compose meaning while text appears.

This is the killer demo: **watch a 3D scene build a sentence, token by token.**

### B.6 Package + Deploy (Day 7)

**Hosting:** Vercel or Netlify (free tier, static site)
**Assets:** ~25MB total (20MB model + 2MB tokenizer + 3MB app code)
**CDN:** Cloudflare for model file caching
**Domain:** sgs-chat.vercel.app (free) or buy sgschat.app ($12/yr)

No backend. No costs beyond the domain name.

---

## Testing

### Phase A (Training)

| What | How | When |
|---|---|---|
| BPE tokenizer | Round-trip: encode → decode = identity | Day 1 |
| Causal masking | Position t only sees tokens <t (no leakage) | Day 2 |
| Forward/backward | Finite logits, non-zero gradients | Day 2 |
| Transmittance caching | Cached = non-cached (same logits) | Day 2 |
| Loss decreases | First 1000 steps | Day 6 |
| Generation coherence | Manual inspection of 20 samples | Day 7 |

### Phase B (Browser)

| What | How | When |
|---|---|---|
| ONNX export correctness | Compare PyTorch vs ONNX outputs on 10 inputs | Day 1 |
| Quantization quality | Compare float32 vs int8 perplexity (should be <1% diff) | Day 1 |
| Browser inference | Same output as Python on same input | Day 2 |
| Tokenizer JS | Same tokenization as Python sentencepiece | Day 3 |
| Cross-browser | Chrome, Firefox, Safari, mobile Chrome | Day 5 |
| Cold start time | Model load < 5 seconds on 4G | Day 7 |
| Token generation speed | > 5 tokens/sec on average laptop CPU | Day 7 |
| Memory usage | < 300MB browser tab | Day 7 |

---

## Monitoring & Telemetry

### In the Browser App

| What | Tool | Privacy |
|---|---|---|
| Page visits, time on site | Plausible Analytics | No cookies, GDPR compliant |
| JS errors, crashes | Sentry (browser) | No PII |
| Model load time | Custom event → Plausible | Aggregated only |
| Tokens generated per session | Custom event → Plausible | Count only, no text |
| Generation speed (tok/sec) | Custom event → Plausible | Aggregated |
| Browser/device/OS | Plausible (built-in) | No fingerprinting |

**What we do NOT collect:** User prompts, generated text, IP addresses, cookies.

### User Feedback

- **"Report Issue" button** → GitHub Issue with template:
  ```
  Browser: [auto-detected]
  Device: [auto-detected]
  Model loaded: [yes/no]
  Generation working: [yes/no]
  What happened: [user fills in]
  ```
- **"Rate this output" thumbs up/down** → anonymous counter (no text stored)
- **GitHub Discussions** for feature requests and general feedback

---

## Rollout

| Stage | Audience | Goal | Duration |
|---|---|---|---|
| **Dev** | Self-testing | Does it work at all? | 2 days |
| **Alpha** | 3-5 ML friends | Catch bugs, test on different devices | 3 days |
| **Beta** | ML Discord (50 users) | Performance on diverse hardware, feedback | 1 week |
| **Launch** | Twitter/X + HN + Reddit r/MachineLearning | Virality | 1 day |
| **Sustain** | Link from papers, GitHub, conferences | Long-tail | Ongoing |

### Launch Assets

- [ ] 30-second screen recording: type prompt → watch generation → show "running locally"
- [ ] Open Graph image: chat interface with Gaussian visualization
- [ ] Thread/post: "We built a language model using 3D rendering math. It runs in your browser for free."
- [ ] Link to paper (A1 or A2)
- [ ] Link to GitHub (full code, reproducible training)

---

## Cost

| Item | Cost |
|---|---|
| TinyStories download | Free |
| Training (RTX 4090, 24h) | ~$10 |
| Vercel hosting | Free tier |
| Domain (optional) | $12/year |
| Plausible Analytics | Free tier (10K views/mo) |
| Sentry | Free tier |
| **Total launch cost** | **~$10-22** |

---

## Parameters

```
Vocabulary:     30,000 × (64 + 64 + 1 + 256) = 11.6M
Position:       256 × 64 = 16.4K
Per-pass MLPs:  ~0.6M
FFN per pass:   ~0.5M
Output proj:    256 × 30,000 = 7.7M
─────────────────────────────────────
Total:          ~20M parameters
On disk:        ~80MB (float32), ~40MB (float16), ~20MB (int8)
```

---

## If It Fails

| Failure | What We Learn | Next Step |
|---|---|---|
| Loss doesn't decrease | Causal rendering can't propagate gradients properly | Debug gradient flow; try non-causal pre-training then causal fine-tune |
| Text is garbage | Random Gaussian init doesn't work for generation | Pre-train vocabulary on word co-occurrence, then fine-tune LM |
| Text is repetitive | Transmittance depletes too fast (later tokens get no weight) | Increase τ; add "transmittance reset" per-pass |
| ONNX export fails | Custom ops not supported | Rewrite rendering equation using only standard ONNX ops |
| Too slow in browser | 20M params too large for WASM | Quantize to int4; reduce d_f to 128 (model shrinks to ~10MB) |
| Browser crashes | Memory too high | Reduce context length to 128; use streaming ONNX |
