# Accelerating SGS Training: Geometry-Aware Losses and Structural Shortcuts

**Status:** working paper, v0.1 (2026-04-20)
**Scope:** proposals to cut Hertz 1B wall-clock training time by exploiting SGS-specific structure (Gaussian tokens, transmittance, multi-pass rendering) rather than generic LM tricks.
**Target:** Planck 1.2 (prototype + validation) and Hertz 1.2 (scale run). Not backported to Hertz 1.0.
**Audience:** internal (Nikita, Claude).

**Strategic context (2026-04-20).** Hertz 1.0 training is paused because the throughput ceiling on a single RTX 4090 (1.7k-11.8k tok/s) makes a 10B-token run a 10-30 day commitment that we do not want to re-run if the architecture changes. Parallel workstreams: finalize Klang (audio), finalize Raum (3D/compositional), validate the blob concept on Planck 1.1, then build this acceleration recipe into Planck 1.2 / Hertz 1.2 before committing to the next large LM run.

---

## 0. Why this paper exists

Hertz 1B trains at 1.7k-11.8k tok/s on a single RTX 4090 depending on config.
Finishing 10B tokens at the midpoint rate is ~10-30 days of continuous
compute. Standard transformer-acceleration recipes (FlashAttention, CUDA
graphs, FSDP) do not directly apply: our hot loop is not QKV-attention but
Gaussian kernel + causal alpha-compositing.

The question this paper addresses: **what accelerations are unlocked by
the fact that we are doing SGS, not attention?**

Five candidate wins, roughly ordered by expected impact and by how much
they change the model vs the training recipe:

1. Transmittance-weighted loss (training signal reshaping).
2. Adaptive pass count (early exit when transmittance saturates).
3. Kernel sparsity (skip far-field Gaussians).
4. Shared kernel across passes (compute once, reuse).
5. Gaussian-native curriculum (order data by transmittance).

The remainder of the paper argues for each, sketches the implementation,
estimates the speedup, and lists risks.

---

## 1. The forward pass, stated precisely

So we have shared language. The model `SGSLanguageModel` in
`src/sgs_lm.py` does, per token sequence `x ∈ ℕ^L`:

### 1.1 Gaussianization (embedding)

Each vocab token `v` owns four parameters:
- `μ_v ∈ ℝ^d_s` (d_s = 256): mean in splatting space.
- `log σ²_v ∈ ℝ^d_s`: diagonal log-variance.
- `raw_α_v ∈ ℝ`: pre-sigmoid opacity.
- `f_v ∈ ℝ^d_f` (d_f = 5000): the feature that gets rendered.

On a forward, tokens become Gaussians via index-lookup:
`μ, log_var, α, features = embed(x)`. Position embedding is added to μ
(additive modulation, not separate dimensions). That is the full "token →
Gaussian" step. There is no further learned projection before rendering.

### 1.2 Pattern assignment: the Gaussian kernel

For queries `q_t` and key Gaussians `(μ_j, Σ_j)`, the kernel is a
Mahalanobis-distance Gaussian:

```
K[t, j] = exp( -½ · (q_t - μ_j)ᵀ Σ_j⁻¹ (q_t - μ_j) / τ )
```

with a learned scalar temperature `τ` (initialized to 128, drifts toward
~23 during training , more on that later). The factored form in
`_pairwise_kernel` avoids materializing the `[B, L, L, d_s]` diff tensor
by expanding the square into three `[B, L, L]` bmms. This is the main
compute per pass.

**This replaces `softmax(Qᵀ K / √d)`.** Importantly: K is **not row-
normalized**. Instead, normalization happens through transmittance.

### 1.3 Rendering: causal alpha-compositing

```
eff_α[t, j] = α_j · K[t, j]               # effective opacity
T[t, j]     = Π_{k < j} (1 - eff_α[t, k])  # transmittance
w[t, j]     = eff_α[t, j] · T[t, j]        # blending weight
meaning[t]  = Σ_j w[t, j] · f_j            # rendered output
```

Transmittance is computed numerically via log-cumsum for stability. The
causal mask is applied to eff_α, zeroing out future tokens before the
cumsum.

**What transmittance means, intuitively.** It is the fraction of "light"
that has survived being absorbed by closer tokens. In the text domain: at
position t, after reading tokens 0..j-1, how much signal is left? If the
model has already absorbed enough context to predict t+1 from tokens
0..j-1, then `T[t, j]` should already be close to 0 (remaining tokens
are redundant).

### 1.4 Multi-pass refinement

Three rendering passes. Between passes, the rendered meaning is used to
update the underlying Gaussians:

```
mu       ← mu + mu_update(features || meaning)       # tanh MLP
alpha    ← alpha * alpha_gate(features || meaning)   # sigmoid MLP
features ← features + pass_ffn(features || meaning)  # 4x GELU MLP (57% of params)
```

So pass 1 renders from raw embeddings; pass 2 renders from refined
Gaussians; pass 3 from twice-refined Gaussians. The FFN here is doing
most of the representational work , by parameter count, 57% of the 1B
model.

### 1.5 Output

`logits = lm_head(LayerNorm(meaning_pass3))`. Standard tied-head projection
to vocab. The SGS-specific machinery stops at `meaning`; the vocab head
is ordinary.

---

## 2. Proposed accelerations

Each proposal is scored on:
- **Est. speedup:** my rough guess, wall-clock.
- **Loss impact:** does it change the training objective / convergence.
- **Implementation cost:** engineering days.
- **Risk:** what could go wrong.

### 2.1 Transmittance-weighted loss

**Insight.** At position t, after the final rendering pass, the model has
produced a transmittance profile `T[t, :]` over keys 0..t. If the model
has confidently absorbed context, `T[t, t]` (transmittance *after* the
last visible token) is near 0. If the model hasn't, `T[t, t]` is near 1.

This is a **per-token confidence signal that is free** , already computed
for the forward. Current loss ignores it entirely:
`loss_t = -log p(x_{t+1} | context)`.

**Proposal.** Modify the per-token loss to

```
loss_t = (1 - T[t, t])^γ · (-log p(x_{t+1} | context))
    + λ · regularizer(T[t, t])
```

where:
- `(1 - T[t, t])^γ` is the **confidence weight** (the model is claiming
  it absorbed the context; so train harder when it's confident AND wrong).
  γ ∈ [1, 2] is a focal-loss-like focusing parameter.
- `regularizer(T)` prevents the trivial collapse `T ≡ 1` (absorb nothing,
  zero loss weight, zero gradient). Candidate: **floor regularizer**
  `max(0, T - T_max)^2` , penalize transmittance above a cap, forcing the
  model to absorb at least *some* signal from context.
- λ is small (≈ 0.01) and annealed down during training.

**Why "transmittance delta below 0" is basically right.** Your framing:
"train data has transmittance 0 (fully absorbed), output has some
residual, delta should always be below 0." Equivalent: we want the model
to bring transmittance down from 1 to ~0 over the course of reading a
sequence. The loss above penalizes it for failing to absorb (regularizer)
and rewards it for confident-and-correct predictions (focal term).

**Est. speedup.** Not a direct tok/s win, but a sample-efficiency win.
Focal loss typically cuts tokens-to-target-loss by 20-40% on language
modeling. If Hertz reaches its target perplexity on 6B tokens instead of
10B, that's a 1.67x wall-clock speedup with **zero kernel changes**.

**Loss impact.** Changes the objective. Needs care: confidence-weighting
can destabilize early training (when model is random, (1-T) is ~uniform
and acts as noise on gradients). Warmup by starting γ = 0 and ramping to
γ = 1 over the first 500M tokens.

**Implementation cost.** ~1 day. Expose `T_final` from the last pass's
`_causal_render`, plumb it through `forward`, add the weighted loss
variant behind a flag `--transmittance-loss`.

**Risk.** Low. Fall back to plain CE if it destabilizes. The regularizer
is the real safety belt , without it the model learns `T ≡ 1` and zeros
out its own loss.

### 2.2 Adaptive pass count (early exit)

**Insight.** We run 3 rendering passes unconditionally. But if after
pass 1 the transmittance profile is already close to its final shape,
passes 2 and 3 are refinement noise. Early in training, the model can't
do anything useful with multi-pass refinement anyway , the Gaussians
haven't learned enough structure for "updated" Gaussians to add signal.

**Proposal.** During training, track the per-pass transmittance profile.
If `||T_pass2 - T_pass1||_∞ < ε`, skip pass 3 for this step. At the batch
level, decide dynamically. Early in training: effectively always run 1
pass. Late in training: always run 3.

**Est. speedup.** 1.5x to 2x during the first 30% of training. Overall
wall-clock: ~1.3x.

**Loss impact.** Unclear. Pass 3 might be a late-training feature that
the model learns to use only once the first two passes are saturated;
in which case this proposal just *avoids* compute that wasn't helping yet.
But it might also be that pass 3 is always needed for the signal to
propagate; in which case early-exit hurts final quality.

**Implementation cost.** ~2 days. The cumsum is not easily differentiable
under variable-pass training (different samples in a batch would have
different computational graphs). Cleanest: per-batch decision, not
per-sample.

**Risk.** Medium. Could silently hurt final perplexity in ways that only
show up at the end of training.

### 2.3 Kernel sparsity (top-k keys per query)

**Insight.** For most queries, only a small fraction of keys have
meaningful kernel values. `K[t, j]` decays as `exp(-d²/τ)`. With the
learned τ ≈ 23 observed mid-training, kernel values at d² > 5τ are below
e^-2.5 ≈ 0.08, effectively noise. These far-field keys contribute
almost nothing to `meaning[t]` but still cost full compute.

**Proposal.** Approximate the dense kernel with a top-k selection per
query:

```
K̃[t, j] = K[t, j]  if j ∈ top-k(distance(q_t, μ_·))
        = 0       otherwise
```

The transmittance cumsum is then over a sparse set. This is **exactly
the 3D Gaussian splatting trick** , tile culling before rendering.

**Est. speedup.** The kernel and rendering together are ~40% of forward
compute. With k=64 at L=512, that's 8x fewer kernel evaluations. Real
win after sparse-matmul overhead: ~2.5x on kernel + render, ~1.4x overall.

**Loss impact.** Controlled. Top-k kernel = known approximation
(Performer / Reformer family). Choose k large enough (k ≥ 64 at L=512,
k ≥ 128 at L=1024) to preserve the >1% tail.

**Implementation cost.** ~3-4 days. Needs a fast top-k over [B, L, L]
tensors. For fp32 this is bandwidth-bound, for bf16 it's cheap. The
cumsum becomes a segmented cumsum , non-trivial but fits in standard
PyTorch.

**Risk.** Medium-high. The first few passes of early training have
uniform kernel (all tokens look the same), so top-k is noise. Enable
only after `opt_step > 5000` or when τ has decayed below some threshold.

### 2.4 Shared kernel across passes

**Insight.** Passes 2 and 3 do `mu ← mu + mu_update(...)`. So the μ
between passes changes *slightly* (the update net is initialized small
and tanh-bounded). The kernel across passes is nearly identical after
early training.

**Proposal.** Compute K once, reuse across all passes. The full SGS
formulation becomes:
- Pass 1: render with K.
- Between passes: update μ, α, features.
- Pass 2, 3: render with the *same* K (but different α, f).

**Est. speedup.** Kernel is ~15% of forward. Computing once vs 3 times:
saves ~10% of forward, ~5% overall.

**Loss impact.** Changes the mathematical model. Passes 2 and 3 no longer
benefit from refined Gaussians in the *kernel* , only in the features.
Plausibly fine; features are what gets composed into meaning. μ-update
then becomes a regularizer pushing toward geometry that looks good *if*
we re-rendered, but never actually re-uses.

**Implementation cost.** ~1 day.

**Risk.** Low-medium. If this hurts final quality, revert in one line.

### 2.5 Gaussian-native curriculum

**Insight.** In image Gaussian splatting, training sorts examples by how
well they've already been reconstructed (loss-sorted curriculum). We can
do the same: sort the FineWeb-Edu shards by **average final
transmittance**. Sequences where the model has already learned to absorb
(low T) get lower sample weight; sequences with high T (surprise) get
higher weight.

**Proposal.** Before each epoch, compute per-sample `mean_t T[t, t]` for
a held-out slice. Bucket into easy/medium/hard. Sample hard buckets more
often.

**Est. speedup.** Classic curriculum gives 10-20% sample efficiency at
the cost of more complex data pipeline. ~1.15x wall-clock.

**Loss impact.** Changes the effective training distribution. Risk of
overfitting to hard examples. Canonical remedy: still sample easy buckets
with minimum probability (temperature).

**Implementation cost.** ~2 days. Needs a data pipeline pass that
computes T per sample. For 10B tokens at 1.7k tok/s, that's a full extra
pass over the data unless we use a subset (say, 200M-token sample).

**Risk.** Medium. Curriculum can bite. Start with a small experiment
(Planck 100M, 1B tokens) before applying to Hertz.

---

## 3. Proposed stack + rollout plan

| # | Proposal | Keep for Hertz 1B? | When | Effort |
|---|----------|:---:|---|---|
| 2.1 | Transmittance-weighted loss | **yes** | Next run | 1 day |
| 2.2 | Adaptive pass count | maybe | After 2.1 | 2 days |
| 2.3 | Kernel sparsity | **yes** | After 2.1 validated | 4 days |
| 2.4 | Shared kernel | yes if cheap | After 2.3 | 1 day |
| 2.5 | Curriculum | **no** for Hertz | Future | 2+ days |

**Recommended order.** 2.1 first because it's cheap, doesn't change
compute shape, and gives a free sample-efficiency win. 2.3 second
because it's the biggest compute win and orthogonal to 2.1. 2.2 and
2.4 are bonus wins once the pipeline stabilizes. 2.5 is Hertz-2 material.

**Compound expected speedup** (multiplicative, with conservative
estimates):
- 2.1: 1.4x sample efficiency
- 2.3: 1.4x forward throughput
- Together: ~**2x** wall-clock
- With 2.2 + 2.4: ~**2.5-3x** in the best case

If the current non-compile throughput is ~12k tok/s, 2x = 24k tok/s,
which would complete 10B tokens in ~5 days instead of 10.

---

## 4. What we are explicitly NOT proposing

- **Dropping to a smaller model (500M).** This is a fallback if the
  above fails. The research claim is "SGS scales to 1B"; the baseline
  is transformer LMs at 1B+ params. Dropping changes the story from
  "does SGS scale?" to "does small SGS work?" , weaker.
- **Mixed-precision beyond bf16.** fp16 has worse range, fp8 is not
  supported on 4090 Inductor paths.
- **FlashAttention.** Doesn't apply , our hot op is not softmax(QKᵀ).
  Analogous work (Flash-Gaussian-Splat) exists for rendering but is not
  a drop-in.
- **Distributed training.** Single-4090 constraint is self-imposed for
  research parity with small-model training. Worth revisiting if we
  get a second GPU.

---

## 5. Open questions

1. **Is transmittance a good confidence signal, empirically?** Verify
   before building 2.1. Quick experiment: in a checkpoint, plot
   `T[t, t]` vs `cross_entropy(prediction, label)`. If they're
   correlated, 2.1 is well-founded.
2. **How fast does τ decay?** Observed 128 → 23 in ~22k opt steps.
   If it keeps dropping, kernel values concentrate more tightly →
   kernel-sparsity wins grow over training. Worth logging `tau.item()`
   in wandb over the full run to confirm.
3. **Does the μ-update between passes actually help?** Ablation: train
   a variant with identity μ-update (still running the MLP but its
   output is ignored). If loss curves match, the μ-update is dead
   weight and 2.4 is free. If they diverge, we keep separate kernels.
4. **What's the right `T_max` in the floor regularizer of 2.1?**
   If too aggressive, model can't represent "I don't know." If too
   lax, model learns T ≡ 1 trivial collapse. Start at T_max = 0.3,
   sweep.

---

## 6. Concrete next actions

1. Verify question 1 above (correlation between T and correctness)
   using the current `checkpoints/hertz/best.pt`. Half-day script.
2. If correlated: implement 2.1 behind `--transmittance-loss`, run a
   500M-token ablation vs plain CE.
3. If ablation shows ≥20% perplexity improvement at equal tokens,
   switch Hertz to it for the remaining training.
4. Concurrently: scope 2.3 (kernel sparsity) , prototype on Planck,
   measure speedup and quality.
5. Re-baseline: re-run `train_hertz.py --no-compile` for 100M tokens
   with nothing added, to confirm the 11.8k tok/s ceiling is stable
   and not a cold-cache mirage.

---

## Appendix A: Why "tokenization, not gaussianization" is a red herring

User raised: "shouldn't we do gaussianization instead of tokenization?"

The tokenizer (BPE, SentencePiece) is an upstream byte-to-integer
compression. It is orthogonal to the SGS formulation. What matters for
SGS is what happens *after* the integer: specifically, the embedding
lookup that produces `(μ, log σ², α, f)` per token. That embedding IS
the gaussianization , tokenization merely decides the granularity at
which the gaussianization is applied.

Alternative: byte-level or char-level SGS would remove BPE and apply the
Gaussian lookup to every byte. This is legitimate research (e.g.,
Canine, ByT5 for transformers) but:
- L grows ~4x → kernel cost grows ~16x.
- The model has to learn BPE-like compression internally in the first
  few layers.
- Net: slower training, weaker quality per compute, same final capability.

BPE stays. The gaussianization is already there, just hidden behind
`nn.Embedding(V, d_s)` and friends.

---

## Appendix B: Why transmittance makes the loss idea defensible

Your intuition: "the training data is what should end up with
transmittance 0; the loss should be the remaining transmittance after
processing."

Framed inside SGS, this is **exactly** the rendering equation's
"remaining light" semantic. In 3D Gaussian splatting, unused Gaussians
pay an opacity-regularization cost , we want *all* useful density to be
captured by the Gaussians. In SGS for LMs, the analog is: we want the
model's cumulative absorption of context to be ~1 at the end of a
sequence (no signal left unmodeled), and the places where absorption
fails are the places where training should push hardest.

The proposal in §2.1 operationalizes this. It is not a different loss
*metric* , it's still cross-entropy , but **weighted by the model's own
self-assessed confidence**, grounded in geometry. That is the SGS-native
version of focal loss.
