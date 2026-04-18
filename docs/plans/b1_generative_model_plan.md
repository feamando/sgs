# B1: Generative SGS Model — Plan (v4)

## Concept

Train a language model that generates text using SGS rendering, then deploy as a **browser-based app** that runs entirely on the user's device. No server, no API key, no cloud.

**The hook:** *"A 100M-parameter language model using 3D rendering math instead of transformers. Trained for $10. Runs in your browser."*

## Two Variants

| Variant | Params | Purpose | Deploy |
|---|---|---|---|
| **B1** | ~100M | Public demo, browser app | ONNX in browser |
| **B1-1** | ~1B | Internal benchmark, push the limits | Python/PyTorch, local |

---

## Two Phases

```
Phase A (Build + Train):  Python/PyTorch, train on RTX 4090
Phase B (Deploy):         Export to ONNX, run in browser via WebGPU/ONNX Runtime
```

---

## Why Edge/Browser Works

| Property | SGS LM (B1) | GPT-2 Small | Llama 3 8B |
|---|---|---|---|
| Parameters | ~100M | 117M | 8B |
| Size on disk (int8) | ~105MB | ~500MB | ~16GB |
| RAM needed | ~300MB | ~1GB | ~32GB |
| CPU inference | ~100ms/token | ~200ms/token | Impossible |
| Runs in browser | **Yes** | Barely | No |

100M params at int8 = ~105MB. Loads in ~5 seconds on broadband. Comparable to GPT-2 Small in size, directly benchmarkable.

---

## Architecture

### B1 (100M — browser demo)

```
SGSLanguageModel(
  vocab: 32K BPE tokens as Gaussians
  d_s: 128 (splatting space)
  d_f: 512 (feature space)
  n_passes: 3
  context: 512 tokens
  n_heads: 4 (multi-head rendering)
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

### A.5 Train (Days 6-7 — GPU)

```bash
python scripts/train_lm.py \
  --data data/tinystories \
  --d_s 128 --d_f 512 --n_passes 3 --n_heads 4 \
  --batch_size 32 --lr 3e-4 --epochs 3 \
  --context_len 512
```

6-12 hours on RTX 4090. ~$10 electricity.

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

| Item | B1 (100M) | B1-1 (1B) |
|---|---|---|
| Training data | TinyStories (free) | FineWeb-Edu (~50GB download) |
| Training time (RTX 4090) | ~12 hours | ~3-5 days |
| Training cost (electricity) | ~$10 | ~$30-50 |
| Vercel hosting | Free tier | N/A (internal) |
| Domain (optional) | $12/year | N/A |
| Plausible / Sentry | Free tier | N/A |
| **Total** | **~$10-22** | **~$30-50** |

---

## Parameters

### B1 (100M — browser demo)

```
Vocabulary:     32,000 × (128 + 128 + 1 + 512) = 24.6M
Position:       512 × 128 = 65.5K
Query proj:     4 heads × (128 × 128 + 128) = 66K
Output proj:    4 × 512 → 512 → 32,000 = 16.9M
Per-pass MLPs:  2 × ~2.5M = 5.0M
Per-pass FFN:   2 × (512+512)×2048×2 = 8.4M
LayerNorms:     ~0.1M
───────────────────────────────────────────────
Total:          ~105M parameters
On disk:        ~420MB (float32), ~210MB (float16), ~105MB (int8)
Browser:        ~105MB download (int8), loads in ~5s on broadband
```

Comparable to GPT-2 Small (117M). Direct head-to-head benchmark possible.

### B1-1 (1B — internal benchmark)

```
Vocabulary:     32,000 × (256 + 256 + 1 + 1024) = 49.2M
Position:       1024 × 256 = 262K
Query proj:     8 heads × (256 × 256 + 256) = 526K
Output proj:    8 × 1024 → 1024 → 32,000 = 41.0M
Per-pass MLPs:  4 × ~8M = 32M
Per-pass FFN:   4 × (1024+1024)×4096×2 = 134M
LayerNorms:     ~0.5M
───────────────────────────────────────────────
Total:          ~1.05B parameters
On disk:        ~4.2GB (float32), ~2.1GB (float16)
VRAM:           ~12GB (mixed precision) — fits on 4090 (24GB)
```

Comparable to TinyLlama (1.1B), Phi-1 (1.3B). Meaningful benchmark territory.
```

---

---

## B1-1: 1B Internal Benchmark

### Why

The 100M model answers "does SGS generate text?" The 1B model answers "can SGS scale?" — a fundamentally different question. At 1B params, we enter the territory of TinyLlama, Phi-1, and OLMo-1B, where meaningful perplexity comparisons are possible.

### Architecture Differences from B1

Updated post-implementation (commit 19c93d0) to fit in 24GB VRAM on a single RTX 4090.
The original plan (d_f=1024, n_passes=5, n_heads=8, context=1024) OOM'd; the current
configuration trades width for rendering passes and attention heads to keep kernel
matrices small while preserving the ~1B parameter budget.

| Property | B1 (100M, shipped) | B1-1 (1B, current) | B1-1 (original plan) |
|---|---|---|---|
| d_s (splatting) | 128 | 256 | 256 |
| d_f (features) | 1000 | 5000 | 1024 |
| n_passes | 3 | 3 | 5 |
| n_heads | 4 | 4 | 8 |
| Context | 512 | 512 | 1024 |
| Vocab | 32K | 32K | 32K |
| Params | ~100M | ~1.04B | ~1.05B |

Rationale for the changes:
- Wider `d_f` concentrates params in dense matmuls (fast on tensor cores) rather than
  in kernel compute.
- Halving heads + passes quarters the kernel matmul cost; with gradient checkpointing
  this is what actually fits on 24GB.
- Shorter context (512) reduces the `[B, L, L]` kernel matrix by 4x. FineWeb-Edu
  documents are truncated to 512 tokens, which is acceptable for perplexity benchmarks.

### Training Data

TinyStories (~2.1M stories) is too small for 1B params — will overfit immediately. Options:

| Dataset | Size | Quality |
|---|---|---|
| **FineWeb-Edu (sample)** | 10B tokens (subset) | High-quality web text, filtered |
| **SlimPajama (sample)** | 10B tokens (subset) | Diverse: web, code, books, wiki |
| **OpenWebText2** | 17B tokens | Reddit-filtered web text |

**Recommendation:** FineWeb-Edu 10B sample. High quality, proven at this scale.

### Training Plan

Current values (actual, post-VRAM tuning); original plan shown for reference.

| Parameter | Current | Original plan |
|---|---|---|
| Data | FineWeb-Edu, 10B tokens | FineWeb-Edu, 10B tokens |
| Micro-batch | 2 | 16 |
| Grad accumulation | 32 (effective 64) | 4 (effective 64) |
| Sequence length | 512 | 1024 |
| Learning rate | 3e-4 (2K warmup) | 1e-4 (with warmup) |
| Training tokens | 10B (1 epoch) | 10B (1 epoch) |
| Estimated wall clock | ~30-60 days on RTX 4090 | 3-5 days |
| Mixed precision | bf16 | bf16 |
| VRAM | ~22GB peak (grad checkpointing ON) | ~12GB peak |

The wall-clock blowup (3-5d → 30-60d) is the cost of fitting 1B params on a single
4090 via gradient checkpointing + reduced micro-batch. Observed throughput is
1.7-2.8k tok/s; 10B tokens / 2.5k tok/s ≈ 46 days. If this is prohibitive,
scope options are: (a) train on 2B tokens (~10 days, weaker LM but same architecture
story), (b) rent a bigger GPU for ~$200-400 on vast.ai/Lambda, (c) shrink `d_f` to
3000 to ~640M params with ~2x throughput.

### What We Measure

| Metric | Compare To |
|---|---|
| Perplexity (FineWeb-Edu val) | TinyLlama-1.1B, Pythia-1B, OLMo-1B |
| HellaSwag (0-shot) | Standard LM benchmark |
| ARC-Easy (0-shot) | Standard LM benchmark |
| Generation quality | Manual eval, side-by-side with TinyLlama |

### What We Learn

| If SGS-1B perplexity is... | Interpretation |
|---|---|
| Within 10% of TinyLlama-1B | **SGS scales.** The rendering equation doesn't limit model quality at scale. Major finding. |
| 10-30% worse | **SGS scales partially.** The composition mechanism introduces some overhead but isn't catastrophic. Worth investigating where the gap is. |
| >30% worse | **SGS has a scaling ceiling.** The rendering equation's constraints (local composition, transmittance depletion) hurt at scale. Would redirect to hybrid architectures. |
| Better than TinyLlama-1B | **Unlikely but transformative.** Would mean rendering-based composition is strictly superior. Paper upgrades to major conference. |

### Risks

1. **Memory:** 1B params at bf16 = 2GB model + activations. Fits on a 4090 24GB only with gradient checkpointing + micro-batch 2.
2. **Speed:** Multi-pass rendering with 3 passes over 512 tokens is ~1.7-2.8k tok/s under torch.compile + bf16. 10B tokens = ~30-60 days wall clock on a single 4090. The main risk is the run straddling a hardware or environment change.
3. **Data download:** FineWeb-Edu sample is ~50GB compressed. Need sufficient disk space.
4. **Convergence:** No one has trained a 1B SGS model before. Hyperparameters may need tuning. Budget for 2-3 restarts.

### Timeline

| Day | What |
|---|---|
| 1 | Download FineWeb-Edu, set up data pipeline |
| 2-3 | Scale SGS model to 1B, verify forward/backward on 4090 |
| 4-8 | Train (3-5 days continuous GPU) |
| 9 | Evaluate: perplexity, benchmarks, generation samples |
| 10 | Compare to TinyLlama/Pythia, write up findings |

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
