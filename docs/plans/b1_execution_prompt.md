# B1 Execution Prompt — For a New Terminal Session

Copy the section below into a fresh Claude Code session in the `~/Documents/GitHub/sgs` directory.

---

## PROMPT

I'm building Radiance Planck — a 100M-parameter language model that uses Semantic Gaussian Splatting (SGS) rendering instead of transformer attention.

**Read these files first for full context:**
- `docs/plans/b1_generative_model_plan.md` — full plan (architecture, training, deployment, risks)
- `docs/plans/roadmap.md` — project overview and where B1 fits
- `docs/brand.md` — naming (Planck = 100M micro model)
- `docs/analysis/phase3_results.md` — latest experiment results
- `src/model.py` — existing SGS encoder (reference for rendering equation implementation)
- `src/kernel.py` — Gaussian kernel (reuse this)
- `src/rendering.py` — alpha-compositing rendering equation (reuse this)

The full project is at `~/Documents/GitHub/sgs`.

### Context

SGS replaces softmax attention with alpha-compositing from 3D Gaussian Splatting. We've proven this is strictly more expressive than softmax (Lean 4 verified) and shown empirical advantages in zero-shot/few-shot settings.

Existing code in the repo:
- `src/kernel.py` — Gaussian kernel evaluation
- `src/rendering.py` — Alpha-compositing rendering equation
- `src/gaussian.py` — Gaussian vocabulary
- `src/model.py` — SGS encoder (sentence similarity, tested)
- `src/seq2seq.py` — SGS seq2seq (SCAN, tested)

### What to Build

A **causal language model** using SGS rendering, trained on TinyStories, deployable in browser.

**Architecture (Radiance Planck — 100M params):**
```
vocab_size: 32,000 (BPE via sentencepiece)
d_s: 128 (splatting space)
d_f: 512 (feature space)
n_passes: 3
n_heads: 4 (multi-head rendering)
context_length: 512
```

**Key implementation details:**
- Causal: position t only sees tokens 0..t-1
- Cache transmittance across positions (T_t = T_{t-1} × (1 - a_{t-1}))
- Multi-pass: 3 passes with MLP position update (tanh bounded), alpha gate (sigmoid), FFN feature update
- Output: meaning vector → linear projection → vocab logits
- Training: next-token prediction (cross-entropy), AdamW, cosine schedule
- Generation: autoregressive with temperature + top-k sampling

**Files to create:**
1. `src/sgs_lm.py` — The causal SGS language model
2. `src/tinystories.py` — TinyStories download + BPE tokenizer + dataloader
3. `scripts/train_lm.py` — Training script with wandb logging
4. `scripts/generate.py` — Generation/sampling script
5. `scripts/export_onnx.py` — ONNX export for browser deployment

**Training data:** TinyStories (HuggingFace `roneneldan/TinyStories`)

**Training config:**
```
batch_size: 32
learning_rate: 3e-4
weight_decay: 0.1
epochs: 3 (over TinyStories)
grad_clip: 1.0
scheduler: cosine with warmup (1000 steps)
context_length: 512
```

**The training will run on a Windows machine with RTX 4090 (24GB VRAM).** Make sure all code works on Windows + CUDA.

### What NOT to do
- Don't modify existing src/ files (kernel.py, rendering.py, etc.)
- Don't use HuggingFace datasets library (SSL issues on Windows — download directly)
- Don't use the GloVe vocabulary — this model learns from scratch via BPE

### Parameter Budget
```
Vocabulary: 32,000 × (128 + 128 + 1 + 512) = 24.6M
Position:   512 × 128 = 65K
Query proj: 4 × (128×128 + 128) = 66K
Output:     4×512 → 512 → 32,000 = 16.9M
Per-pass:   3 × ~3M = 9M
Total:      ~105M
```

### Testing Checklist
After building, verify:
- [ ] Forward pass produces finite logits (no NaN)
- [ ] Backward pass: all parameters receive gradients
- [ ] Causal masking: position t cannot see token t or later
- [ ] Generation produces varied tokens (not all same ID)
- [ ] Training loss decreases over first 100 steps
- [ ] Memory fits in 24GB VRAM with batch_size=32

Start by building `src/sgs_lm.py` and `src/tinystories.py`, then the training script. Test each component as you go.

---

## Execution Order

```
Day 1: src/tinystories.py — download + tokenizer + dataloader
Day 2: src/sgs_lm.py — causal SGS language model
Day 3: Smoke tests — forward, backward, causal masking, memory
Day 4: scripts/train_lm.py — training loop with logging
Day 5: scripts/generate.py — sampling + quality check
Day 6: Train on RTX 4090 (~12h)
Day 7: Evaluate — perplexity, samples, compare to transformer baseline
```
