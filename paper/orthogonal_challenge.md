# Paper Orthogonal Challenge

**Challenger Role:** Hostile reviewer at a top NLP venue
**Date:** 2026-04-09

---

## Overall Assessment

The paper presents an interesting cross-domain transfer (3DGS → NLP) with a strong theoretical contribution (Theorem 1) and an eye-catching SCAN result. However, several claims are overstated, key confounds are unaddressed, and the experimental methodology has gaps that a careful reviewer would exploit. 10 challenges below.

---

## CRITICAL — Could lead to rejection

### C1: The SCAN Result Is From a Single Seed With No Error Bars

The 45.7% vs 0.0% headline is from seed=42 only. No standard deviation. No multiple runs. SCAN is known to have high variance across random seeds — some seeds produce near-zero for all architectures, others produce moderate results.

Lake & Baroni (2018) and subsequent work (Csordas et al., 2021; Kim & Linzen, 2020) report SCAN results averaged over 5+ seeds. A single run is not publishable evidence.

**What a reviewer would say:** "The central claim rests on a single random seed. The SCAN length split is notoriously unstable. Please report mean ± std over at least 5 seeds."

**Required fix:** Run SCAN length 5 seeds. Report mean ± std. If the mean is >20% it's still strong. If variance is huge (e.g., 45% ± 40%), the result is unreliable.

---

### C2: The Transformer Baseline for SCAN Is Weak

The paper compares SGS (355K params, GRU decoder) against a vanilla Transformer (963K params, standard encoder-decoder). But the SCAN literature has moved far beyond vanilla transformers:

- **Relative positional encoding** transformers reach ~70-90% on length split (Csordas et al., 2021)
- **Data augmentation** approaches reach ~99% (Andreas, 2020)
- **Specialized architectures** (CGPS, NQG) reach ~100% (Li et al., 2019; Shaw et al., 2021)

The claim "SGS 45.7% vs Transformer 0.0%" is misleading because the transformer baseline is the weakest possible variant. A reviewer will immediately ask: "How does SGS compare to a transformer with relative positional encoding? Or a CGPS baseline?"

**What a reviewer would say:** "The transformer baseline lacks relative positional encoding, which is known to dramatically improve SCAN length generalization. The comparison is unfair."

**Required fix:** Add a transformer with relative positional encoding (sinusoidal or RoPE). If SGS still beats it, the result is stronger. If not, the claim needs scoping: "SGS provides compositional inductive bias comparable to positional encoding innovations."

---

### C3: The SCAN Encoder and STS-B Encoder Are Different Architectures

A subtle but important issue: the SCAN SGS encoder (`SGSSeq2Seq` in `src/seq2seq.py`) uses **learned embeddings** — it does NOT use the Gaussian vocabulary with PCA-initialized means, covariances, and opacity from GloVe. It uses fresh `nn.Embedding` layers for both splatting positions and features.

This means the SCAN result tests "does alpha-compositing with a GRU decoder help on SCAN?" — not "does the full SGS architecture (Gaussian vocabulary + rendering equation) help?" The Gaussian primitive with covariance and opacity (Atom A1) is largely absent from the SCAN experiment. The rendering equation (Atom A3) is present, but operating on simple learned embeddings, not Gaussians.

**What a reviewer would say:** "The SCAN encoder uses standard embeddings, not Gaussian distributions. The contribution of the Gaussian primitive vs. the rendering equation is confounded."

**Required fix:** Acknowledge this explicitly. The SCAN result validates the rendering equation (A3 — alpha-compositing with transmittance), not the full Gaussian primitive (A1 — covariance, opacity, kernel). This is still valuable but the claim should be scoped correctly.

---

### C4: The Theorem Doesn't Imply Better Performance

Theorem 1 proves alpha-compositing is more expressive than softmax. But the paper implies this explains the empirical advantages. It doesn't — more expressive doesn't mean better.

ReLU networks are universal approximators (more expressive than linear models), but linear regression often outperforms deep networks on small tabular datasets. Expressiveness is about the set of representable functions; performance depends on inductive bias, optimization landscape, and data.

The empirical advantage of SGS comes from its **inductive bias** (structural composition, locality, ordering), not from expressiveness per se. The theorem is a nice theoretical result but should not be presented as causally linked to the empirical gains.

**What a reviewer would say:** "The theorem proves expressiveness, but the experiments show an inductive bias advantage. These are different claims. The connection between Theorem 1 and the experimental results is unclear."

**Required fix:** Decouple the theorem from the empirical claims. Present Theorem 1 as an independent theoretical contribution. Present the empirical results as evidence of inductive bias advantages. They support each other but one doesn't prove the other.

---

## MAJOR — Would require revision

### C5: "Matched Architecture" Is Not Truly Matched

The paper claims "Fair Softmax" is a matched architecture comparison. But:

- SGS has learned Gaussian parameters (covariance, opacity) that softmax doesn't have
- The rendering equation uses the kernel (a complex nonlinear function of position) while softmax uses dot-product scores (a simpler function)
- SGS has a learned temperature τ; softmax uses √d scaling

These are all differences beyond just "rendering vs softmax." The true matched comparison would hold ALL other variables constant and ONLY change the weight computation. As implemented, SGS has more representational capacity in the splatting space.

**Required fix:** Acknowledge the architectural differences. The comparison isolates "rendering-based composition vs. attention-based composition" but not "rendering equation vs. softmax" in isolation.

---

### C6: The "Complementary Mechanisms" Framing Is Unfalsifiable

The paper positions SGS and softmax as "complementary" — SGS for composition, softmax for distribution. This is a convenient framing that cannot be wrong: whatever result you get, you can claim it supports complementarity.

If SGS wins → "rendering is better for this task"
If softmax wins → "attention is better for this task"
If they tie → "they're equivalent, hence complementary"

A strong paper would make a falsifiable prediction: "SGS should win on X and lose on Y, for specific reasons Z." The current framing is post-hoc narrative.

**Required fix:** Either make specific, falsifiable predictions about which tasks SGS should win on (and test them), or drop the complementarity framing and just report the results.

---

### C7: Zero-Shot Results Conflate Initialization With Architecture

The zero-shot results (SGS 0.707 vs softmax 0.609) are presented as evidence of SGS's compositional advantage. But in zero-shot mode, the ONLY difference is:

- SGS: Gaussian kernel weighting (proximity in PCA space) with transmittance
- Softmax: dot-product weighting (angle in PCA space) without transmittance

The "advantage" could be entirely due to the **kernel function** (Gaussian RBF vs dot-product), not the **composition mechanism** (rendering vs. attention). Gaussian kernels are known to outperform dot-product similarity in many settings.

**Required fix:** Add a "Gaussian kernel attention" baseline — softmax over Gaussian kernel values instead of dot products: `weights = softmax(K(q, μ_i, Σ_i))`. If this matches SGS zero-shot, the advantage is the kernel, not the rendering equation.

---

## MODERATE — Would strengthen the paper

### C8: No Comparison to Other Compositional Generalization Methods

The SCAN section compares only SGS vs vanilla Transformer. The literature includes many approaches specifically designed for compositional generalization:

- LANE (Liu et al., 2020): lexical attention network
- CGPS (Li et al., 2019): compositional generalization via primitives
- Meta-learning approaches (Lake, 2019)
- Least-to-most prompting (Drozdov et al., 2022) — mentioned but not compared

At minimum, the paper should discuss how SGS's 45.7% compares to these methods and acknowledge which ones SGS does or doesn't beat.

---

### C9: The Paper Doesn't Address Negation/Quantification

The original whitepaper identified negation and quantification as key challenges (operator Gaussians). The paper silently drops this — there's no mention of the limitation that alpha-compositing is monotonic and can't negate.

**Required fix:** Add a Limitations section acknowledging that SGS handles ~85% of language (monotonic composition) and that non-monotonic composition (negation, quantification, scope) requires extensions not yet tested.

---

### C10: Missing Confidence Intervals on STS-B Phase 3 Results

Phase 3 STS-B results (SGS 0.726 vs softmax 0.729) are from a single seed. The claimed "convergence" could be a coincidence. Phase 1.5 used 3 seeds — why not Phase 3?

---

## Summary of Required Fixes

| Priority | Fix | Effort |
|---|---|---|
| **P0** | Run SCAN 5 seeds, report mean ± std | 2-3 hours GPU |
| **P1** | Add transformer with relative positional encoding to SCAN | 1 day |
| **P2** | Acknowledge SCAN uses learned embeddings, not full Gaussian primitive | Text edit |
| **P3** | Decouple Theorem 1 from empirical claims | Text edit |
| **P4** | Add "Gaussian kernel attention" zero-shot baseline | 2 hours |
| **P5** | Add Limitations section (negation, monotonic-only) | Text edit |
| **P6** | Run Phase 3 STS-B with 3 seeds | 4 hours GPU |
| **P7** | Discuss SCAN in context of existing compositional methods | Text edit |
| **P8** | Scope "matched architecture" claims more carefully | Text edit |
| **P9** | Remove or weaken "complementary" framing | Text edit |

**Bottom line:** The paper has strong ingredients (theorem, SCAN result, comprehensive experiments) but needs tightening. The biggest risk is C1+C2: if the SCAN result doesn't replicate across seeds or collapses with a better transformer baseline, the central empirical claim falls apart. Fix those first.
