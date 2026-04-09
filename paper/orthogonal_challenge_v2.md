# Paper v2 Orthogonal Challenge

**Challenger Role:** Third-round reviewer, looking for remaining gaps
**Date:** 2026-04-09

---

## Assessment

The revised paper is substantially more honest than v1. The SCAN claim is properly scoped, the Limitations section addresses key gaps, the theorem is decoupled from empirical claims, and the ablation revealing the GRU decoder as the primary driver is prominently featured. This version would survive most hostile reviews.

Remaining issues are moderate, not critical.

---

## MODERATE ISSUES

### M1: The Core Contribution Is Narrower Than the Paper Suggests

After all the caveats, what has been demonstrated?

1. A theorem about weight vector expressiveness (Lean verified) — strong, novel
2. Alpha-compositing provides better zero-shot and few-shot sentence similarity than softmax (+0.08 / +0.027) — real but modest
3. At scale, the two converge — so the practical advantage is limited to low-data regimes
4. On SCAN, the GRU decoder matters more than the encoder — SGS is not uniquely advantaged

The honest summary: "We found a theoretically interesting alternative to softmax that is marginally better with limited data and equivalent at scale." This is a solid workshop paper or short paper, but the current framing still suggests more than the evidence supports.

**Suggestion:** Consider whether the paper would be stronger as a focused short paper: Theorem 1 + zero-shot/few-shot advantage + honest SCAN ablation. Drop the "complementary mechanisms" narrative and let the results speak.

---

### M2: Zero-Shot Advantage May Be the Kernel, Not the Rendering Equation

Challenge C7 from the first review was not fully addressed. The zero-shot advantage (SGS 0.707 vs softmax 0.609) could be due to the Gaussian kernel function rather than the alpha-compositing mechanism. To isolate:

- "Gaussian Kernel Softmax" = softmax over kernel values: $w_i = \text{softmax}(\mathcal{K}(q, \mu_i, \Sigma_i))$

If this matches SGS zero-shot, the advantage is the kernel, not the rendering equation. This is a quick experiment (~1 hour) that would strengthen or weaken the core claim. Not running it leaves an obvious gap.

---

### M3: The "Self-Sufficiency" Claim Needs Context

The paper presents "IDF/PC1 removal hurt" as evidence that SGS is self-sufficient. But this was tested only on STS-B with 5.7K pairs. On a different task or with different data, IDF init might help. The finding is specific to one setting, not a general property of the architecture.

**Suggestion:** Scope the self-sufficiency claim to "on STS-B with GloVe initialization."

---

### M4: Still No Comparison to SBERT/SimCSE

The paper compares SGS to mean-pool and softmax attention on Gaussian vocabularies. It does not compare to actual sentence embedding methods (SBERT, SimCSE) that people use in practice. A reviewer will ask: "Why would I use SGS instead of SimCSE?"

The answer is presumably "SGS is a composition mechanism, not a production model" — but this should be stated. The paper's contribution is the mechanism and the theory, not a state-of-the-art sentence embedding.

**Suggestion:** Add a sentence in the introduction: "We do not aim to compete with production sentence embedding systems (Reimers & Gurevych, 2019; Gao et al., 2021). Our contribution is a novel composition mechanism with provable properties and a characterization of its inductive bias relative to softmax attention."

---

### M5: The Training Stability Claim Is Under-Evidenced

The paper claims "SGS has the tightest error bars (±0.0017)." But this was measured on one task (STS-B) with one training recipe. Fair Softmax had ±0.0089 — 5× wider but still small. The claim would be stronger with stability measured across multiple tasks and hyperparameter settings.

---

### M6: Future Work / Hybrid Architecture Not Explored

The paper suggests "combining SGS's inductive bias with softmax's capacity" as future work. This is the obvious next step, and not even a preliminary experiment was run. A single experiment — SGS rendering for pass 1, softmax attention for pass 2 — would take hours and could demonstrate whether the hybrid works.

---

## What the Paper Gets Right (Strengths)

1. **The theorem is the strongest contribution.** A formally verified (Lean 4) result connecting computer graphics and NLP. Novel, clean, publishable on its own.

2. **The SCAN ablation is honest and informative.** Most papers would report only SGS vs Transformer. This paper runs 5 models × 5 seeds and honestly reports that the GRU decoder is the primary driver. This builds trust.

3. **The inductive bias framing is correct.** The CNN vs ViT analogy is apt and the data supports it: strong bias helps early, flexibility wins late.

4. **Negative results are reported.** IDF init, PC1 removal, multi-head — all hurt. This is valuable information presented transparently.

5. **Comprehensive verification.** 13 Lean 4 proofs, 85+ literature review, multi-seed experiments. The methodology is rigorous.

---

## Verdict

**The paper is publishable as-is for a workshop or short paper.** For a main conference (EMNLP/NeurIPS), the contribution is at the boundary — the theorem is strong but the empirical novelty is modest (marginal improvement in low-data, equivalence at scale, SCAN ablation reduces the headline result).

The paper would be strengthened by:
1. Running the Gaussian Kernel Softmax baseline (M2) — 1 hour
2. Scoping claims more tightly (M1, M3) — text edits
3. Adding the SBERT/SimCSE positioning statement (M4) — 1 sentence
4. A preliminary SGS+attention hybrid experiment (M6) — could strengthen to full paper

**No critical issues remain.** The paper is honest about what it shows and what it doesn't.
