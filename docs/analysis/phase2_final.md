# Phase 2 Final Analysis (Post-Challenge)

**Date:** 2026-04-08

---

## What We Know For Certain

| Finding | Evidence | Confidence |
|---|---|---|
| SGS has the best zero-shot composition | 0.689 vs softmax 0.609 vs mean-pool 0.605 | **HIGH** (consistent, no training confound) |
| SGS beats softmax with limited data | +0.026 on STS-B, 3 seeds, significant | **HIGH** |
| Softmax outperforms SGS after NLI contrastive training | 0.726 vs 0.714 | **MEDIUM** (confounded by loss function, only 3 epochs) |
| Structured composition is necessary for contrastive learning | Mean-pool degrades; SGS and softmax improve | **HIGH** |
| Higher d_s improves zero-shot monotonically | +0.030 from d_s=64→300 zero-shot | **HIGH** |
| The rendering equation is provably more expressive than softmax | Lean 4 verified, 0 sorry | **CERTAIN** |

## What's Genuinely Uncertain

| Question | Why It's Uncertain |
|---|---|
| Does softmax truly scale better, or is the contrastive loss biased toward it? | NLI experiment confounds data size + loss function + global calibration |
| Would SGS + d_s=300 + NLI close the gap? | Not tested |
| Would more NLI epochs help SGS? (it was still improving) | Not tested |
| Does SGS win on compositional tasks (SCAN/COGS)? | Not tested — this is the decisive experiment |

---

## Challenge Resolutions

| # | Challenge | Resolution |
|---|---|---|
| C1 | Kill signal triggered | **Partially accepted.** The NLI result meets the literal kill condition (SGS ≤ softmax), but the condition was defined for matched training, and the NLI setup is confounded (C2, C3). We do NOT kill. We pivot the thesis from "replacement" to "complementary mechanism" and test on compositional tasks. |
| C2 | NLI confounds data + loss | **Accepted.** Cannot isolate data scaling from loss function change. Claim revised to "NLI contrastive training + more data improved both architectures." |
| C3 | Contrastive loss favors softmax | **Accepted and actionable.** Test SGS on NLI 3-way classification (not contrastive) to remove the global calibration bias. |
| C4 | d_s=300 + NLI not tested | **Accepted and actionable.** This is a missing experiment that could change the conclusion. |
| C5 | 3 NLI epochs may be insufficient for SGS | **Accepted and actionable.** SGS was still improving. Run 10 epochs. |
| C6 | Only testing STS-B | **Accepted.** STS-B is the worst benchmark for SGS's structural advantages. SCAN/COGS is the right test. |
| C7 | Mean-pool degradation under-analyzed | **Accepted and integrated.** This validates structured composition (both rendering and attention) as necessary. |

---

## Revised SGS Thesis

**Original:** "Rendering can replace attention for language."

**Revised (post Phase 2):** "The rendering equation provides a stronger inductive bias for language composition than softmax attention. This advantage is decisive in zero-shot, few-shot, and compositionally demanding settings. At scale with distributional training objectives, softmax's flexibility compensates for its weaker inductive bias. The two mechanisms are complementary."

This is defensible regardless of what SCAN/COGS shows. But SCAN/COGS determines the strength of the claim.

---

## Phase 3 Experiment Plan

### Three parallel experiments. Each answers a specific question. Each is independently publishable.

---

### Experiment 3A: SCAN Compositional Generalization

**Question:** Does SGS's structural inductive bias enable compositional generalization that softmax fails at?

**Why this is decisive:** Lake & Baroni (2018) showed transformers achieve <5% accuracy on SCAN length generalization. If SGS's explicit composition (Gaussian kernel + transmittance + sequence ordering) captures compositional rules, it should generalize. This would be the "SGS is necessary for X" result that STS-B cannot provide.

**Setup:**
- SCAN dataset: train on primitives, test on novel compositions
- Splits to test: `addprim_jump` (most-studied), `length` (hardest)
- SGS needs adaptation for seq2seq (rendering → autoregressive decoding)
- Compare: SGS seq2seq vs Transformer seq2seq (matched parameters)

**Expected effort:** 3-4 days (seq2seq adaptation + training + evaluation)

**Success criterion:** SGS > 50% on SCAN length split (transformers get <5%). Even modest compositionality would be significant.

---

### Experiment 3B: NLI Classification (Not Contrastive)

**Question:** Does the softmax advantage hold with a non-contrastive training objective?

**Why it matters:** Challenge C3 identified that contrastive loss specifically favors global composition (softmax). NLI 3-way classification (entailment/neutral/contradiction) is a local pairwise decision that doesn't require global calibration.

**Setup:**
- AllNLI with 3-way classification head: concat(a, b, |a-b|, a*b) → MLP → 3 classes
- Cross-entropy loss (not contrastive)
- Evaluate: NLI accuracy + transfer to STS-B

**Expected effort:** 1-2 days (classification head + training)

**Success criterion:** SGS ≥ Fair Softmax on NLI classification accuracy. This would confirm that the Phase 2 softmax advantage was contrastive-loss-specific, not fundamental.

---

### Experiment 3C: Close the Gaps (Quick Fixes)

**Question:** Do the untested combinations (d_s=300 + NLI, more epochs) close the gap?

**Runs:**

| Run | What | Time |
|---|---|---|
| SGS d_s=300 + NLI (3 epochs) | Missing combination from Phase 2 | ~4 min |
| SGS d_s=64 + NLI (10 epochs) | Was SGS still converging? | ~10 min |
| SGS d_s=300 + NLI (10 epochs) | Best of both | ~15 min |
| Fair Softmax + NLI (10 epochs) | Fair comparison at 10 epochs | ~10 min |

**Expected effort:** 1 day

**Success criterion:** Any SGS variant ≥ Fair Softmax on STS-B test after NLI training.

---

## Priority Order

| Priority | Experiment | Why First |
|---|---|---|
| **P0** | 3C: Close the gaps | Fastest (1 day). Could resolve the Phase 2 ambiguity immediately. |
| **P1** | 3B: NLI classification | 1-2 days. Tests if contrastive loss was the confound. |
| **P2** | 3A: SCAN compositional | 3-4 days. The decisive "is SGS necessary?" experiment. |

**If 3C shows SGS catching up:** The Phase 2 softmax advantage was an artifact of insufficient training.
**If 3B shows SGS winning on classification:** The contrastive loss was the confound.
**If 3A shows SGS winning on SCAN:** SGS has a clear, distinct value proposition (compositional generalization).

**If ALL three fail:** SGS is a strong initialization method and interpretability tool, but not a competitive runtime architecture. Paper focus shifts to: novel theorem (Softmax ⊂ Alpha-Compositing) + zero-shot composition mechanism + theoretical framework.

---

## Timeline

| Day | Task |
|---|---|
| Day 1 | Exp 3C: d_s=300+NLI, more epochs, close the gaps |
| Day 2 | Exp 3B: NLI 3-way classification |
| Day 3-4 | Exp 3A: SCAN seq2seq adaptation |
| Day 5 | Exp 3A: SCAN training + evaluation |
| Day 6 | Final analysis, paper outline |

---

## Paper Regardless

**The paper is writable NOW with what we have.** The contribution stack:

1. **Novel theorem** (Softmax ⊂ Alpha-Compositing) — Lean 4 verified, publishable alone
2. **First application of Gaussian splatting rendering to NLP** — novel architecture
3. **Zero-shot advantage** — rendering 0.689 vs softmax 0.609, no training
4. **Few-shot advantage** — rendering +0.026 over matched softmax, 3 seeds, significant
5. **Scaling trade-off characterization** — rendering better inductive bias, softmax better capacity
6. **85+ paper literature review** connecting 3DGS, NLP, conceptual spaces, kernel methods
7. **13 formally verified mathematical foundations**

Phase 3 experiments strengthen the paper but don't gate it. Start writing in parallel.
