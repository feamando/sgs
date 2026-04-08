# Phase 1 Results — Orthogonal Challenge

**Challenger Role:** Adversarial reviewer
**Date:** 2026-04-08
**Target:** Phase 1 Results Analysis v1

---

## Challenge Summary

The analysis presents SGS results in a favorable light, but several claims are weaker than stated. The experiment has methodological issues that inflate the apparent advantage of SGS over baselines. Below are 8 challenges.

---

## CRITICAL CHALLENGES

### C1: The Softmax Attention Baseline Is Unfairly Weak

**The claim:** "Alpha-compositing beats softmax attention by +0.077."

**The problem:** The softmax attention baseline (`SoftmaxAttentionModel`) is a single-layer, single-head attention with query = mean of means. This is not a fair comparison to SGS-2pass, which has multi-pass refinement with learned MLPs, position embeddings, and a learned temperature.

A fair comparison would be:
- **Softmax baseline with the SAME number of parameters** (add FFN layers, position embeddings, etc.)
- **Softmax baseline with multi-layer stacking** (2 layers of attention, matching SGS-2pass)
- **Softmax baseline with learned query** (not just centroid)

As-is, the comparison is "rich SGS model (22M params, position embeddings, learned τ, multi-pass MLPs) vs. bare softmax on raw Gaussians (21.5M params, no position embeddings, no learned components beyond vocabulary)." The 0.5M extra parameters in SGS include the position embeddings and per-pass MLPs that the softmax baseline lacks.

**Required resolution:** Run a softmax baseline with identical architecture components (position embeddings, 2 layers of attention + FFN, learned temperature). The delta should be measured at matched architecture complexity, not matched vocabulary.

---

### C2: The Val-Test Gap Undermines All Comparisons

**The claim:** Various comparative deltas between models.

**The problem:** The val-test gap is 0.10-0.14 for all models. This means the training set (5.7K pairs) is too small to reliably distinguish models that differ by 0.04-0.08 in test performance. The ranking could be partially driven by which model happens to overfit in a way that transfers slightly better to the test distribution.

Specifically:
- SGS-2pass: val 0.7615, test 0.6580 (gap: 0.104)
- Mean-pool features: val 0.7560, test 0.6159 (gap: 0.140)
- The val scores are nearly tied (0.76 vs 0.76) but test scores differ by 0.04

This suggests the test delta is driven by generalization luck, not systematic architectural advantage. A single seed (42) is insufficient to establish significance.

**Required resolution:** Run 5 seeds and report mean ± std. If the confidence intervals overlap, the comparison is not statistically significant.

---

### C3: The "0.78 Target" Is the Wrong Benchmark

**The claim:** "SGS must reach 0.78 (SIF) to pass."

**The problem:** SIF (Arora et al., 2017) is not a comparable baseline. SIF uses:
1. IDF weighting (frequency-based importance — SGS doesn't have)
2. Common component removal (subtracts the first principal component — SGS doesn't do)
3. No training at all (it's unsupervised)

SIF's 0.78 comes from these two tricks on top of GloVe, not from a superior composition mechanism. If you add IDF weighting to mean-pooling (which is trivial), you'd get ~0.72 — much closer to SGS's 0.66.

The correct comparison is SGS (0.6580) vs. trained mean-pool (0.6159) = **+0.042 delta on matched conditions.** This is the real signal. Comparing to SIF is apples-to-oranges because SIF has domain-specific tricks that SGS hasn't incorporated.

**Required resolution:** Implement SIF as a baseline within the experiment (add IDF weighting and common component removal). Then compare SGS to SIF directly.

---

## MAJOR CHALLENGES

### C4: PCA at 44.7% Variance Is Crippling the Model

**The claim:** "The dual-space architecture works."

**The problem:** Only 44.7% of GloVe variance is preserved in the 64d splatting space. More than half the semantic information is discarded before the rendering equation even runs. The rendering equation is operating on a severely degraded representation.

This means we're not testing "does rendering work for language?" — we're testing "does rendering work on a lossy compression of language?" The answer might be "rendering is great but 44.7% variance is too little" rather than "rendering is only OK."

**Required resolution:** Run with d_s=128 (should capture ~60-65% variance) and d_s=300 (full GloVe, no PCA). If performance improves substantially with more dimensions, the d_s=64 result is not representative of the architecture's potential.

---

### C5: The Query Mechanism Is Trivially Bad

**The claim:** "SGS demonstrates a clear advantage over mean-pooling."

**The problem:** The query is `mean(μ)` — the centroid of all word positions. This is literally a bag-of-words signal. The entire multi-pass, transmittance-gated, Gaussian-kernel rendering equation is being asked to do its work from a single, undifferentiated query point that contains no information about WHICH aspect of meaning to extract.

It's like pointing a camera at the center of a room and asking "what do you see?" You see everything blurred together. The power of the rendering equation (viewpoint-dependent composition) is completely unused.

The fact that SGS STILL beats mean-pooling despite this crippled query is arguably more impressive than if it worked perfectly — but the comparison is testing the rendering equation in its worst possible operating mode.

**Required resolution:** Implement multi-head viewpoints (Atom A4) with learned projection matrices. This is the mechanism that actually differentiates SGS from averaging — and it hasn't been tested yet.

---

### C6: The 8-Pass Failure Is Concerning

**The claim:** "2 passes is the sweet spot; 8 passes overfits."

**The problem:** The analysis dismisses the 8-pass failure as "overfitting on small data." But this interpretation is convenient, not proven. An alternative explanation: the multi-pass update mechanism (MLP_μ, MLP_α, FFN) is unstable — each pass amplifies noise rather than refining meaning. The fact that 4 passes shows no improvement over 2 (0.6578 vs 0.6580) suggests the multi-pass mechanism is hitting diminishing returns very early, not that it's "working but saturating."

If multi-pass rendering is supposed to be the analog of transformer depth (12-96 layers), and it breaks at 4 passes on trivial data, this is a red flag for scaling.

**Required resolution:** Analyze what's happening in passes 3-8. Are μ values diverging? Is α collapsing to zero? Are features degenerating? Log per-pass metrics to diagnose whether this is data-size overfitting or architectural instability.

---

## MODERATE CHALLENGES

### C7: Zero-Shot Results Tell an Interesting Story

**The claim:** Focuses mainly on trained results.

**The problem:** The zero-shot (pre-training) results are under-analyzed:

| Model | Zero-shot Val |
|---|---|
| No transmittance | **0.7012** |
| SGS-2pass | 0.6885 |
| SGS-1pass | 0.6808 |
| SGS-full (P=4) | 0.6775 |
| Mean-pool features | 0.6045 |
| Mean-pool means (μ) | 0.5775 |
| Softmax attention | 0.5237 |

Before ANY training, SGS with transmittance already achieves 0.70 — better than trained softmax attention (0.58)! And "no transmittance" achieves the highest zero-shot score (0.70). This suggests the GloVe initialization + kernel weighting is doing most of the work, and training is actually degrading the generalization.

This is a strong result that the analysis under-sells. It means the Gaussian representation + kernel composition captures meaningful similarity structure from GloVe alone.

**Required resolution:** Highlight and analyze zero-shot results. The 0.70 zero-shot with rendering vs. 0.60 mean-pool is a +0.10 delta with NO training — this is a cleaner signal than trained comparisons.

---

### C8: No Error Bars, No Statistical Testing

**The claim:** All comparative deltas.

**The problem:** Every number is from a single seed. With 5.7K training pairs and 1.4K test pairs, the standard error on Spearman is approximately ±0.02-0.03. Several of the claimed deltas (e.g., SGS vs mean-pool at +0.042) are within this margin.

**Required resolution:** Run at minimum 3 seeds (ideally 5) and report mean ± std. Apply a paired bootstrap test for the key comparisons.

---

## ALTERNATIVE INTERPRETATIONS

### Alt-1: The Kernel Is Doing All the Work, Not the Rendering Equation

The kernel evaluation K(q, μ, Σ) assigns different weights to different words based on their proximity to the query in splatting space. This is itself a form of attention — just Gaussian-kernel attention instead of dot-product attention. The transmittance is a minor add-on (+0.035). The multi-pass is saturated at 2.

Interpretation: **It's the Gaussian kernel (Atom A2) that matters, not the compositing equation (Atom A3).** The "rendering equation" frame is mostly the kernel, with transmittance and ordering being marginal contributors.

### Alt-2: GloVe Is the Real Hero

The zero-shot results (0.70 for rendering, 0.60 for mean-pool) show that GloVe initialization contributes a +0.10 delta via kernel weighting. Training adds another +0.05 at best. The rendering equation is a good way to query GloVe embeddings, but it's the embeddings — not the composition mechanism — driving performance.

### Alt-3: STS-B Is Too Easy/Small to Differentiate Architectures

STS-B has 5.7K training pairs. Modern models train on millions. All models overfit (val-test gap 0.10+). The true test of SGS's composition mechanism needs a larger dataset where composition genuinely matters (e.g., NLI, paraphrase detection, or downstream tasks with complex sentences).

---

## Verdict

The Phase 1 results are **promising but methodologically incomplete.** The rendering equation shows a real advantage, but the magnitude of that advantage is uncertain due to:
1. Unfair baselines (softmax without matched architecture)
2. Single seed (no error bars)
3. Crippled query mechanism (centroid)
4. Severe information loss (44.7% PCA)

The strongest finding is actually the **zero-shot result**: rendering achieves 0.70 vs. 0.60 for mean-pooling WITHOUT any training. This is a clean signal that doesn't depend on optimization dynamics or overfitting patterns.

**Recommendation:** Before declaring Phase 1 complete, address C1 (fair softmax baseline), C4 (higher d_s), C5 (learned query), and C8 (multiple seeds). These are all implementable in days, not weeks.
