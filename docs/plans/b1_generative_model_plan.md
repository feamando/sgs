# B1: Tiny Generative SGS Model — Plan (v2)

## Concept

Train a small language model that generates text using SGS rendering instead of transformer attention. Not aiming for GPT quality — aiming for "it works, it's novel, it cost $X."

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

## Training Data

**TinyStories** (Eldan & Li, 2023)
- ~2.1M short children's stories
- Simple vocabulary, short sentences
- Proven: 30M-param models generate readable stories on this data
- Download: HuggingFace `roneneldan/TinyStories`

---

## Implementation Phases

### Phase 1: Data Pipeline (Day 1)

```python
# src/tinystories.py
1. Download TinyStories from HuggingFace
2. Train BPE tokenizer (30K vocab) via sentencepiece or tiktoken
3. Tokenize all stories → binary format for fast loading
4. DataLoader: random crops of 256 tokens, causal (left-to-right)
```

### Phase 2: Causal SGS Model (Days 2-3)

```python
# src/sgs_lm.py
class SGSLanguageModel(nn.Module):
    """Autoregressive language model using SGS rendering."""

    def __init__(self, vocab_size=30000, d_s=64, d_f=256, n_passes=2, max_len=256):
        # Gaussian vocabulary (learned from scratch, not GloVe)
        self.mu = nn.Embedding(vocab_size, d_s)
        self.log_var = nn.Embedding(vocab_size, d_s)
        self.raw_alpha = nn.Embedding(vocab_size, 1)
        self.features = nn.Embedding(vocab_size, d_f)

        # Position embeddings
        self.pos_mu = nn.Embedding(max_len, d_s)
        self.pos_alpha = nn.Embedding(max_len, 1)

        # Per-pass updates (causal)
        self.mu_update = nn.ModuleList([...])
        self.alpha_gate = nn.ModuleList([...])
        self.ffn = nn.ModuleList([...])

        # Output head
        self.log_tau = nn.Parameter(...)
        self.out_proj = nn.Linear(d_f, vocab_size)  # or tie with features

    def forward(self, token_ids):
        """
        Causal language modeling.
        For each position t, render meaning from tokens 0..t-1.
        Return logits for each position.
        """
        batch, seq_len = token_ids.shape
        # ... activate Gaussians, apply positional modulation ...

        all_logits = []
        for t in range(1, seq_len):
            # Render from tokens 0..t-1 (causal)
            meaning = render_causal(gaussians[:, :t, :], query_t)
            logits_t = self.out_proj(meaning)
            all_logits.append(logits_t)

        return torch.stack(all_logits, dim=1)  # [batch, seq-1, vocab]
```

**Key challenge:** Naive causal rendering is O(n²) — rendering at position t requires compositing t Gaussians. For n=256, that's 256 rendering passes. Optimizations:
- **Cache transmittance:** T_t = T_{t-1} × (1 - a_{t-1}). Don't recompute from scratch.
- **Vectorized rendering:** Compute all positions in parallel using a causal mask (like transformer causal attention).
- **Chunked rendering:** Process in chunks of 32 tokens.

### Phase 3: Training Loop (Day 4)

```python
# scripts/train_lm.py
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
scheduler = CosineAnnealingLR(T_max=num_steps)

for batch in dataloader:
    logits = model(batch)                    # [batch, seq-1, vocab]
    targets = batch[:, 1:]                   # shift right
    loss = cross_entropy(logits, targets)    # next-token prediction
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

### Phase 4: Generation (Day 5)

```python
# src/generate.py
def generate(model, prompt_ids, max_new_tokens=200, temperature=0.8, top_k=50):
    for _ in range(max_new_tokens):
        logits = model(prompt_ids)[:, -1, :]  # last position
        logits = logits / temperature
        # Top-k filtering
        top_k_logits, top_k_indices = logits.topk(top_k)
        probs = softmax(top_k_logits)
        next_token = top_k_indices[torch.multinomial(probs, 1)]
        prompt_ids = torch.cat([prompt_ids, next_token], dim=1)
    return prompt_ids
```

### Phase 5: Train (Day 6 — GPU day)

```bash
python scripts/train_lm.py \
  --data data/tinystories \
  --vocab_size 30000 \
  --d_s 64 --d_f 256 \
  --n_passes 2 \
  --batch_size 32 \
  --lr 3e-4 \
  --epochs 3 \
  --max_len 256
```

Estimated: 12-24 hours on RTX 4090.

### Phase 6: Evaluate + Compare (Day 7)

- **Perplexity** on TinyStories validation set
- **Sample quality:** Generate 100 stories, manual inspection
- **Baseline comparison:** Train a matched-param transformer LM on the same data
- **The question:** Does SGS text make sense? Is it qualitatively different from transformer text?

---

## Testing

| What | How | When |
|---|---|---|
| **BPE tokenizer** | Round-trip: encode → decode should be identity | Phase 1 |
| **Causal masking** | Verify position t only sees tokens <t (no leakage) | Phase 2 |
| **Forward pass** | Smoke test: random input → finite logits, no NaN | Phase 2 |
| **Backward pass** | All parameters receive non-zero gradients | Phase 2 |
| **Caching** | Cached rendering matches non-cached (same logits) | Phase 2 |
| **Training loss** | Loss decreases over first 1000 steps | Phase 5 |
| **Generation** | Model doesn't repeat or degenerate (check diversity) | Phase 6 |
| **Perplexity** | Compare to matched transformer | Phase 6 |

---

## Monitoring & Telemetry

### During Training

| Metric | Logging | Frequency |
|---|---|---|
| Training loss | wandb or tensorboard | Every 10 steps |
| Validation loss / perplexity | wandb | Every 500 steps |
| Learning rate | wandb | Every step |
| Gradient norm | wandb | Every 100 steps |
| Learned τ (temperature) | wandb | Every 100 steps |
| α distribution (mean, std, min, max) | wandb | Every 500 steps |
| μ norm distribution | wandb | Every 500 steps |
| Generation samples | wandb text | Every 1000 steps |
| GPU memory / utilization | nvidia-smi log | Continuous |
| Training throughput (tokens/sec) | wandb | Every 100 steps |

### Checkpointing

- Save every 2000 steps
- Keep best 3 by validation loss
- Save optimizer state for resume
- Export best model weights separately for inference

### Post-Training Evaluation

| Metric | How |
|---|---|
| Perplexity | Standard next-token loss on held-out set |
| BLEU / ROUGE (vs references) | Not applicable for generative LM |
| Repetition rate | % of 4-grams that repeat within 200 tokens |
| Distinct-n | Ratio of unique n-grams (measures diversity) |
| Human eval | Manual rating of 50 samples: coherence (1-5), grammar (1-5) |
| Vs transformer baseline | Same metrics, same data, same param count |

---

## Rollout

This is internal/exploratory. No public launch.

| Stage | What | Decision Gate |
|---|---|---|
| **Build** | Implement model + training | Can it train (loss decreasing)? |
| **Train** | Run on TinyStories | Does loss converge? |
| **Evaluate** | Perplexity + samples | Is text readable? |
| **Compare** | Vs matched transformer | Is SGS competitive? |
| **Demo** | Internal Jupyter notebook with generation | Worth showing externally? |
| **Share** | If quality is decent: blog post or A3 visualizer integration | — |

### If It Fails

If the generative model produces garbage: that's an informative negative result. "Alpha-compositing does not natively support autoregressive text generation" would be worth documenting. Possible reasons:
1. Causal rendering can't build sufficient context (transmittance depletes too fast)
2. The Gaussian vocabulary needs pre-training (random init doesn't work for generation)
3. The query mechanism needs to be position-specific, not centroid-based

Each failure mode points to a specific fix.

---

## Cost

| Item | Cost |
|---|---|
| TinyStories download | Free |
| Training (12-24h on RTX 4090) | ~$5-10 electricity |
| wandb | Free tier |
| **Total** | **~$10** |

---

## Parameters

```
Vocabulary:     30,000 × (64 + 64 + 1 + 256) = 11.6M
Position:       256 × (64 + 1) = 16.6K
Per-pass MLPs:  ~0.6M × 1 pass = 0.6M
FFN per pass:   ~0.5M × 1 = 0.5M
Output proj:    256 × 30,000 = 7.7M
─────────────────────────────────────
Total:          ~20M parameters
```

Comparable to GPT-2 small (117M) at ~1/6 the size. A matched transformer baseline at 20M params would be roughly 4-layer, 4-head, d_model=256.
