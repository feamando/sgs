# Phase 2 Plan: Pushing SGS Toward Competitive Performance

**Date:** 2026-04-08
**Starting point:** SGS-2pass = 0.6756 ± 0.0017 on STS-B
**Target:** ≥ 0.72 on STS-B, competitive on at least one downstream task

---

## What's Already Proven

1. **Rendering > Softmax Attention** (+0.026, 3 seeds, matched architecture, significant)
2. **Rendering > Mean-Pooling** (+0.059, significant)
3. **Transmittance adds signal** (+0.035 over kernel-only)
4. **2-pass refinement helps** (+0.043 over 1-pass)
5. **Architecture is stable** (±0.0017 across seeds — tightest of any model)
6. **SGS is self-sufficient** (external tricks like IDF/PC1 removal hurt — it learns its own weighting)

**We do NOT need to re-prove any of these.** They're settled.

---

## What We Need to Prove Next

There are exactly two open questions:

### Question 1: Is the 0.676 ceiling a DATA problem or an ARCHITECTURE problem?

SGS trains on 5,749 sentence pairs. It has 22M parameters. That's ~4,000 parameters per training example — massive overfitting is guaranteed. The val-test gap (0.757 vs 0.676 = 0.081) confirms this.

**Hypothesis H1:** SGS's absolute performance is bottlenecked by training data, not architecture. With more data, SGS will exceed 0.72.

**Test:** Train SGS-2pass on AllNLI (~500K pairs) instead of STS-B (5.7K). Keep everything else identical.

**If H1 is true:** Performance jumps significantly (to 0.72+). The architecture is fine; we just need more data.

**If H1 is false:** Performance plateaus even with more data. The architecture has a ceiling — probably the query mechanism or the splatting space dimensionality.

### Question 2: Is the 44.7% PCA variance limiting composition quality?

The splatting space (d_s=64) captures only 44.7% of GloVe's variance. The kernel operates in this reduced space, so 55% of the embedding structure is invisible to the composition mechanism. It only affects the kernel weights — features still carry the full 300d signal.

**Hypothesis H2:** Increasing d_s to capture more variance improves the kernel's ability to weight words, which improves composition.

**Test:** Run SGS-2pass at d_s=128 and d_s=300 (no PCA).

**If H2 is true:** Higher d_s improves test Spearman. This means the kernel needs more information to make good weighting decisions.

**If H2 is false:** Higher d_s doesn't help (or hurts due to curse of dimensionality in kernel evaluation). The 64d splatting space captures the relevant structure.

---

## What We Are NOT Testing Yet

- ~~Better query mechanism~~ — Phase 1.5 showed multi-head projections hurt. Before trying more query variants, we need to understand if data or dimensionality is the bottleneck. If more data fixes it, the centroid query is fine.
- ~~Operator Gaussians~~ — STS-B doesn't test negation or quantification.
- ~~Compositional generalization (SCAN/COGS)~~ — need a working seq2seq SGS first.
- ~~Downstream tasks~~ — prove the ceiling is liftable first.

---

## Implementation Plan

### Experiment 1: More Data (Priority 1, ~2 days)

**What:** Train SGS-2pass on AllNLI (SNLI + MultiNLI) using the sentence-transformers NLI training recipe, then evaluate on STS-B.

**How it works:**
- AllNLI has ~557K sentence pairs labeled entailment/neutral/contradiction
- Training objective: contrastive loss (similar sentences close, contradictory sentences far)
- This is how sentence-transformers (SBERT) trains — proven to transfer well to STS-B
- After NLI training, evaluate on STS-B test WITHOUT fine-tuning on STS-B

**Why this matters:** It separates the data question from the architecture question. If SGS reaches 0.72+ when trained on 100x more data, the Phase 1 ceiling was data-limited. If it still plateaus at ~0.68, the architecture needs changes.

**What we compare:**
- SGS-2pass (NLI-trained) → STS-B test
- Mean-pool (NLI-trained) → STS-B test
- Fair Softmax (NLI-trained) → STS-B test
- The deltas should match or exceed Phase 1.5 deltas

**Implementation:**
1. New data loader for AllNLI (entailment/contradiction/neutral labels)
2. Contrastive training loss: `loss = -log(exp(cos(a,p)/τ) / Σ exp(cos(a,n)/τ))`
3. Evaluate on STS-B test zero-shot (no STS-B training)

### Experiment 2: Higher d_s (Priority 2, ~1 day)

**What:** Run SGS-2pass on STS-B with d_s ∈ {32, 128, 300}.

**Implementation:** Already supported in the code (d_s is a config parameter). Just three training runs:

```bash
python scripts/train_stsb.py --glove data/glove.6B.300d.txt --model sgs --n_passes 2 --d_s 32
python scripts/train_stsb.py --glove data/glove.6B.300d.txt --model sgs --n_passes 2 --d_s 128
python scripts/train_stsb.py --glove data/glove.6B.300d.txt --model sgs --n_passes 2 --d_s 300
```

**What we learn:**
- d_s=32: does less variance (fewer dimensions) help or hurt?
- d_s=128: does more variance help? (Expected: ~60-65% explained)
- d_s=300: full GloVe space — no PCA at all. Splatting space = feature space. Removes the dual-space design entirely. If this works best, the architecture is simpler than we thought.

**Concern with d_s=300:** The temperature τ was proven to work at d_s=64. At d_s=300, E[D_M]=300, so τ=300 is needed. The kernel may be less discriminative. Phase 0 showed sparsity is already poor at d=64 — at d=300 it will be worse. But for STS-B (short sentences, n<50), the O(n²) cost doesn't matter.

---

## Decision Matrix

| Outcome | Interpretation | Next Step |
|---|---|---|
| **H1 true + H2 true** | Data AND dimensions both help | Scale up: larger data + optimal d_s → Phase 2 (downstream tasks) |
| **H1 true + H2 false** | Data is the bottleneck, d_s=64 is fine | Keep d_s=64, train on larger data → Phase 2 |
| **H1 false + H2 true** | Need more dimensions, data alone insufficient | Redesign splatting space → possibly non-PCA learned projection |
| **H1 false + H2 false** | Architecture has a ceiling at ~0.68 | The rendering equation works but is limited for STS → test on OTHER tasks where composition matters more (NLI, paraphrase, SCAN) |

**Note on the last case:** A ceiling of 0.68 on STS-B is not a failure. STS-B is a surface-level similarity task. The rendering equation's advantage (explicit composition, transmittance-based ordering) may show up more on tasks that REQUIRE compositional understanding — like NLI ("A dog is running" entails "An animal is moving" but not vice versa) or paraphrase detection ("He gave her a book" = "She received a book from him" — requires understanding argument structure).

---

## Timeline

| Day | Task | GPU Hours |
|---|---|---|
| 1 | Implement AllNLI data loader + contrastive loss | 0 |
| 2 | Train SGS + baselines on AllNLI, eval on STS-B | 8h |
| 3 | Run d_s sweep (32, 128, 300) on STS-B | 3h |
| 4 | Analyze results, write up | 0 |

Total: ~4 days, ~11 GPU hours.

---

## Success Criteria

**Phase 2 passes if ANY of these hold:**
1. SGS on AllNLI achieves STS-B test ≥ 0.72
2. SGS at higher d_s achieves STS-B test ≥ 0.72
3. SGS on AllNLI beats Fair Softmax on AllNLI by ≥ 0.02 (maintaining the advantage at scale)

**Phase 2 triggers pivot if:**
- SGS on AllNLI < 0.68 AND SGS at d_s=300 < 0.68 → architecture ceiling is real
- SGS on AllNLI ≤ Fair Softmax on AllNLI → rendering advantage disappears with more data

**Even if we don't reach 0.72, the existing results are publishable.** The proven Softmax ⊂ Alpha-Compositing theorem + the empirical validation (rendering > attention, significant) is a complete paper. Phase 2 strengthens it but isn't required.
