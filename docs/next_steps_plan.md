# SGS Next Steps Plan

**Date:** 2026-04-08
**Status:** Phase 1 complete (INVESTIGATE). Moving to Phase 1.5 + parallel tracks.
**Repo:** github.com/feamando/sgs

---

## Where We Are

Phase 1 proved the rendering equation works for language composition:
- Zero-shot: rendering 0.70 vs mean-pool 0.60 (+0.10, no training confounds)
- Trained: SGS 0.66 > softmax 0.58 (+0.08)
- Transmittance adds +0.035; 2-pass is optimal
- 13/13 mathematical claims formally verified in Lean 4
- Novel theorem: Softmax ⊂ Alpha-Compositing (proven)

**Not yet proven:** Can SGS reach competitive absolute performance (>0.72)? Are the deltas statistically significant? Does viewpoint-dependent rendering add value?

---

## Plan Overview

```
Week 1:  Phase 1.5 — Fix known bottlenecks, validate results
Week 2:  Phase 1.5 — Learned query, higher d_s, fair baselines
Week 3:  Phase 2a  — Multi-head viewpoints, downstream tasks
Week 4:  Phase 2b  — Compositional generalization (SCAN/COGS)
Week 5+: Decision  — Publish / Scale / Pivot
```

**Parallel track (ongoing):** Write the paper. Enough material exists for a submission regardless of Phase 2 outcomes.

---

## Phase 1.5: Fix Bottlenecks (Weeks 1-2)

The orthogonal challenge identified 5 fixable issues. Each is a code change, not a research question.

### 1.5.1 Multi-Seed Validation (Day 1)

**Goal:** Get error bars on all comparisons.

**Implementation:**
```bash
# Run SGS-2pass + all baselines on 5 seeds
for seed in 42 123 456 789 1337; do
  python scripts/train_stsb.py --glove data/glove.6B.300d.txt --model sgs --n_passes 2 --seed $seed
  python scripts/train_stsb.py --glove data/glove.6B.300d.txt --model mean_pool --seed $seed
  python scripts/train_stsb.py --glove data/glove.6B.300d.txt --model softmax_attn --seed $seed
done
```

**Success:** SGS > mean-pool with non-overlapping 95% CIs. If CIs overlap, the +0.042 delta is not significant.

**Effort:** Modify `run_all_ablations.py` to accept `--seeds` flag. ~4 hours GPU time.

---

### 1.5.2 Learned Query / Multi-Head Viewpoints (Days 2-4)

**Goal:** Replace the crude centroid query with learned projections — the single biggest expected improvement.

**Implementation:** New `MultiHeadSGSEncoder`:

```python
# Per head: learned projection P_h maps mean-centroid to a query
# Then render from that query position
class MultiHeadSGSEncoder(SGSEncoder):
    def __init__(self, ..., n_heads=4):
        self.query_proj = nn.ModuleList([
            nn.Linear(d_s, d_s) for _ in range(n_heads)
        ])
        self.output_proj = nn.Linear(n_heads * d_f, d_f)
```

Each head learns a different "viewpoint" — one might focus on subjects, another on modifiers, another on actions. The concatenated multi-view output should capture richer compositional structure than a single centroid query.

**Ablation:** 1 head vs 2 vs 4 vs 8. Probe heads for specialization (does head 1 attend to nouns? head 2 to verbs?).

**Success:** Multi-head SGS > single-head SGS by ≥ 0.02 Spearman.

**Effort:** ~50 lines of code. 1 day implementation + training.

---

### 1.5.3 IDF Opacity Initialization (Day 2)

**Goal:** Give SGS the same frequency-weighting advantage that SIF has.

**Implementation:**
```python
# In gaussian.py init_from_glove():
# Instead of uniform alpha (sigmoid(0) = 0.5), use IDF
idf = log(N / (1 + doc_freq))  # or approximate from rank
raw_alpha_init = inverse_sigmoid(idf_normalized)
```

Common words ("the", "a", "is") start with low opacity; rare content words start with high opacity. SIF's entire advantage comes from this weighting — giving it to SGS for free should close a significant portion of the gap to 0.78.

**Success:** SGS + IDF init > SGS without by ≥ 0.03.

**Effort:** ~20 lines. Hours.

---

### 1.5.4 Common Component Removal (Day 2)

**Goal:** Remove the dominant direction from the feature space (the other SIF trick).

**Implementation:**
```python
# After PCA, compute first principal component of features
# Subtract its projection from all feature vectors
pc1 = pca.components_[0]
features -= features @ pc1[:, None] @ pc1[None, :]
```

This removes the "average meaning" direction, which inflates cosine similarity for all pairs and compresses the effective similarity range.

**Success:** Combined with IDF, should push toward 0.72+.

**Effort:** ~10 lines.

---

### 1.5.5 Fair Softmax Baseline (Day 3)

**Goal:** Match the softmax baseline to SGS's architecture complexity.

**Implementation:** `FairSoftmaxModel` with:
- Same position embeddings as SGS
- Same learned temperature
- 2 layers of softmax attention + FFN (matching SGS-2pass parameter count)
- Same training recipe

**This is the decisive comparison.** If SGS-2pass still beats matched softmax, the rendering equation genuinely outperforms attention for composition. If matched softmax catches up, the advantage was from position embeddings and MLPs, not from alpha-compositing.

**Effort:** ~100 lines. 1 day.

---

### 1.5.6 Higher Splatting Dimension (Day 4)

**Goal:** Test whether the 44.7% PCA variance is limiting performance.

**Runs:**
- d_s=32 (less variance, but closer to 3DGS's d=3)
- d_s=128 (more variance, ~60-65%)
- d_s=300 (full GloVe, no PCA — splatting = feature space)

**If d_s=300 wins significantly:** The dual-space design may be unnecessary — operating directly in the full embedding space could be simpler and better. This would change the architecture significantly but simplify it.

**Effort:** Config change + 3 training runs. Hours.

---

### 1.5.7 Per-Pass Diagnostics (Day 4)

**Goal:** Understand why 8 passes hurts — is it overfitting, opacity collapse, or μ divergence?

**Implementation:** Add logging to `SGSEncoder.forward()`:
```python
# Log per pass:
# - mean/std of α across tokens
# - mean/std of Δμ (position update magnitude)
# - mean/std of kernel values K
# - feature norm ||f||
# - transmittance T at last position
```

**Effort:** ~30 lines of logging code.

---

## Phase 2a: Viewpoint Specialization (Week 3)

*Only if Phase 1.5 achieves > 0.72 with statistical significance.*

### 2a.1 Multi-Task Viewpoint Training

Train SGS encoder on multiple tasks simultaneously, each with a different viewpoint:

| Task | Dataset | What the viewpoint extracts |
|---|---|---|
| Sentiment | SST-2 | Positive/negative |
| Topic | AG News | News category |
| Similarity | STS-B | Overall meaning |
| NLI | SNLI/MultiNLI | Entailment/contradiction |

Each task gets its own viewpoint projection. Shared Gaussian vocabulary. This tests whether different viewpoints genuinely specialize for different semantic aspects.

**Success:** Per-task viewpoint heads outperform shared single-head on at least 2 tasks.

### 2a.2 Viewpoint Probing

After training, analyze what each head learned:
- Attention-map equivalents: which Gaussians contribute most to each head?
- POS correlation: does head 1 weight nouns? head 2 weight verbs?
- Semantic role correlation: does one head extract agents, another extract patients?

This is the interpretability test — if SGS claims to be more interpretable than transformers, it must demonstrate here.

---

## Phase 2b: Compositional Generalization (Week 4)

*Only if Phase 2a shows viewpoint specialization.*

### 2b.1 SCAN Benchmark

SCAN tests systematic compositionality: train on primitives ("jump", "turn left"), test on novel combinations ("jump twice after turn left").

This is where SGS's explicit composition operator should shine — if the rendering equation captures genuine compositional structure, it should generalize to novel combinations better than attention.

**Implementation:** Adapt SGS for sequence-to-sequence (autoregressive generation via rendering).

### 2b.2 COGS Benchmark

COGS tests compositional generalization for semantic parsing. More linguistically grounded than SCAN.

### 2b.3 Operator Gaussians for Negation

Test operator Gaussians (Atom A6) on:
- Monotonicity NLI (Yanaka et al., 2019)
- NegNLI negation benchmark
- SNLI negation subset

**If operators fail:** Drop them. Handle negation in the FFN within multi-pass. The rendering equation handles monotonic composition (~85% of language); the FFN handles the rest.

---

## Parallel Track: Write the Paper (Ongoing)

Enough material exists NOW for a workshop paper or short paper. The Phase 1 results + Lean proofs + novel theorem are a complete contribution.

### Paper Structure

**Title:** "Semantic Gaussian Splatting: Alpha-Compositing as a Composition Mechanism for Language"

**Sections:**
1. Introduction — the analogy, the hypothesis
2. Background — 3DGS, Word2Gauss, attention
3. Theorem: Softmax ⊂ Alpha-Compositing (with Lean proof)
4. Architecture — SGS encoder (7 atoms)
5. Experiments — STS-B ablations, zero-shot analysis
6. Analysis — what works (kernel, transmittance), what doesn't yet (sparsity, deep passes)
7. Related work (85+ papers surveyed)
8. Conclusion + future work

**Target venues:**
- EMNLP 2026 (deadline ~Jun 2026) — full paper
- NeurIPS 2026 workshop (deadline ~Sep 2026) — if results need more time
- ICLR 2027 (deadline ~Oct 2026) — if Phase 2 results are strong

**What makes it publishable now:**
1. Novel theorem (Softmax ⊂ Alpha-Compositing) — machine-verified
2. First application of Gaussian splatting rendering to pure NLP
3. Empirical evidence that rendering outperforms softmax for composition
4. Comprehensive theoretical framework (7 atoms, 13 proven claims)

---

## Decision Gates

| After | Condition | Decision |
|---|---|---|
| Phase 1.5 (Week 2) | Multi-seed SGS > mean-pool (p<0.05) AND > 0.72 | **PASS** → Phase 2 |
| Phase 1.5 (Week 2) | SGS > matched softmax (fair baseline) | **Rendering validated** |
| Phase 1.5 (Week 2) | SGS ≤ matched softmax despite improvements | **PIVOT** → Gaussian Transformer hybrid |
| Phase 1.5 (Week 2) | SGS < 0.65 despite all improvements | **KILL** → Rendering doesn't compose language |
| Phase 2a (Week 3) | Viewpoint heads specialize | → Phase 2b (compositionality) |
| Phase 2a (Week 3) | No specialization | → Paper with Phase 1 results only |
| Phase 2b (Week 4) | SGS beats transformer on SCAN/COGS | → Major paper (new paradigm claim) |
| Phase 2b (Week 4) | SGS ≈ transformer on SCAN/COGS | → Paper scoped to "alternative composition" |

---

## Resource Allocation

| Task | GPU Hours | Human Days | Priority |
|---|---|---|---|
| 1.5.1 Multi-seed | 4h | 0.5d | P0 |
| 1.5.2 Learned query | 4h | 1d | P1 |
| 1.5.3 IDF init | 1h | 0.5d | P1 |
| 1.5.4 Common component removal | 1h | 0.5d | P1 |
| 1.5.5 Fair softmax baseline | 4h | 1d | P1 |
| 1.5.6 Higher d_s | 3h | 0.5d | P2 |
| 1.5.7 Per-pass diagnostics | 2h | 0.5d | P2 |
| 2a Multi-task viewpoints | 12h | 3d | P3 |
| 2b SCAN/COGS | 8h | 2d | P4 |
| Paper writing | — | 5d | Parallel |
| **Total** | **~39h** | **~15d** | |

All fits within your 4090 in ~2 days of continuous GPU time. The human effort is the bottleneck — ~3 weeks of part-time work.

---

## What Success Looks Like at Each Stage

### After Phase 1.5 (Week 2)
> "SGS-2pass with multi-head queries and IDF init achieves 0.74 ± 0.02 Spearman on STS-B test, statistically significantly above trained mean-pooling (0.62 ± 0.02, p<0.01) and matched softmax attention (0.67 ± 0.02, p<0.05). The rendering equation is validated as a composition mechanism for language."

### After Phase 2a (Week 3)
> "Different viewpoint heads specialize for different semantic aspects: head 1 weights nouns (subject extraction), head 3 weights sentiment-bearing adjectives. Multi-task training with shared Gaussian vocabulary improves per-task performance over single-task by 2-4%."

### After Phase 2b (Week 4)
> "SGS achieves 85% accuracy on SCAN length generalization split, vs. 20% for a matched transformer. The explicit compositional structure of the rendering equation generalizes to novel combinations."

### Paper Submission (Week 5)
> "We introduce Semantic Gaussian Splatting and prove that the alpha-compositing rendering equation from 3D Gaussian Splatting is strictly more expressive than softmax attention (Lean 4 verified). Empirically, SGS outperforms attention-based composition on STS-B and compositional generalization benchmarks, while offering interpretable, per-Gaussian contribution tracing."
