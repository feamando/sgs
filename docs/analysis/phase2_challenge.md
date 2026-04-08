# Phase 2 Results — Orthogonal Challenge

**Challenger Role:** Adversarial reviewer
**Date:** 2026-04-08

---

## Challenge Summary

The Phase 2 analysis frames the results optimistically as a "CNN vs ViT" inductive bias story. But several of the findings undermine the SGS thesis more than acknowledged. 7 challenges below.

---

## CRITICAL CHALLENGES

### C1: The Rendering Advantage Disappeared — This Is the Kill Signal You Defined

**The claim:** "Rendering has stronger inductive bias but softmax scales better — an interesting trade-off."

**The problem:** Go back to the Phase 1.5 analysis. The key comparison was defined as:

> "If SGS > matched softmax → rendering validated. If SGS ≤ matched softmax → pivot to Gaussian Transformer hybrid."

Phase 1.5: SGS 0.676 > Fair Softmax 0.649. Rendering validated.
Phase 2 NLI: SGS 0.714 < Fair Softmax 0.726. **Rendering invalidated by your own criterion.**

The "inductive bias trade-off" framing is a narrative pivot to avoid triggering the kill condition. But the condition was clear: "SGS ≤ matched softmax despite improvements → PIVOT." The NLI experiment IS the improvement (100x more data), and softmax won.

**The honest reading:** The rendering equation's advantage is a small-data artifact. With sufficient training data — which any real application will have — softmax attention is simply better. The Gaussian kernel weighting provides a good initialization (zero-shot 0.689 vs 0.609), but the learned composition mechanism matters more than the inductive bias once you have data.

**Required resolution:** Either (a) acknowledge this as a kill signal and pivot to the hybrid, or (b) define a specific scenario where SGS wins AT SCALE (not just with limited data) and test it. "Better zero-shot" is not a viable long-term thesis for a new architecture.

---

### C2: The NLI Experiment Is Not a Fair Data Scaling Test

**The claim:** "Data is the bottleneck — NLI training jumps SGS from 0.676 to 0.714."

**The problem:** The STS-B and NLI experiments use DIFFERENT training objectives:
- STS-B: MSE on similarity scores (regression)
- NLI: Multiple Negatives Ranking Loss (contrastive)

So the improvement from 0.676 → 0.714 confounds TWO changes: more data AND a different loss function. The contrastive loss is known to produce better sentence embeddings than MSE regression regardless of data size (this is the entire SBERT insight, Reimers & Gurevych, 2019).

To isolate the data effect, you'd need:
- Same loss (contrastive) on STS-B vs AllNLI — but STS-B doesn't have the right labels for contrastive
- Or: same loss (MSE) with more STS-style data — but that doesn't exist at 300K scale

The comparison is inherently confounded. We cannot confidently say "data was the bottleneck" because the loss change alone might account for the improvement.

**Required resolution:** Acknowledge the confound. The correct claim is "NLI contrastive training + more data improved performance." Whether it's the data or the loss is unknown.

---

### C3: Fair Softmax Had an Architectural Advantage in the NLI Setting

**The claim:** "Fair Softmax overtakes SGS with more data."

**The problem:** Look at the NLI improvement magnitudes:
- SGS: +0.038 (0.676 → 0.714)
- Fair Softmax: +0.077 (0.649 → 0.726)

Fair Softmax improved TWICE as much from NLI training. Why?

The contrastive loss computes a similarity matrix between all pairs in a batch. This requires the encoder to produce globally comparable embeddings — any sentence must be comparable to any other. Softmax attention's global all-to-all weighting naturally produces globally comparable representations. SGS's transmittance-gated local composition produces representations that are good for pairwise comparison but may not be as globally calibrated.

In other words: **the contrastive loss specifically favors softmax's global composition over SGS's local composition.** This is not "softmax scales better" — it's "the training objective suits softmax better."

**Required resolution:** Test with a loss function that doesn't require global calibration — e.g., pairwise cosine regression (like STS-B but with more data), or a classification task (NLI 3-way classification instead of contrastive).

---

## MAJOR CHALLENGES

### C4: The d_s Sweep Contradicts the Dual-Space Architecture

**The claim:** "Dimensionality is not the primary bottleneck."

**What the data actually shows:**

| d_s | Zero-shot val | Δ from d_s=64 |
|---|---|---|
| 32 | 0.6732 | -0.015 |
| 64 | 0.6885 | baseline |
| 128 | 0.7015 | +0.013 |
| 300 | 0.7187 | **+0.030** |

Zero-shot performance improves monotonically with d_s, and the gain from 64→300 (+0.030) is LARGER than the gain from 64→128 (+0.013). This means the gains are NOT diminishing in zero-shot mode. The kernel IS better with more information.

After training, the gains shrink because training compensates for the information loss. But the ARCHITECTURE is better at d_s=300 — the training just can't exploit it with 5.7K examples.

**Implication:** The NLI experiment should be re-run at d_s=300 (no PCA). The combination of more data + more dimensions might push SGS above Fair Softmax.

**Required resolution:** Run SGS d_s=300 + NLI. This was not tested and could change the outcome.

---

### C5: 3 Epochs of NLI Is Not Enough for SGS

**The claim:** "Fair Softmax overtakes SGS with more data."

**The data:**
```
SGS     Epoch 1: 0.7987 → Epoch 2: 0.8049 → Epoch 3: 0.8060 (still rising)
Softmax Epoch 1: 0.7948 → Epoch 2: 0.8068 → Epoch 3: 0.8050 (peaked at 2)
```

SGS was still improving at epoch 3 (+0.0011/epoch). Softmax peaked and declined. With more epochs:
- SGS might continue climbing and overtake
- Or SGS might plateau just below softmax

3 epochs is standard for SBERT training with 500K+ examples, but SGS has a different optimization landscape (Gaussian parameters in splatting space have different gradient dynamics than embedding tables). It may need more epochs to converge.

**Required resolution:** Run NLI training for 5-10 epochs and check if SGS catches up.

---

### C6: You're Only Testing One Task (STS-B)

**The claim:** "Softmax scales better."

**The problem:** STS-B tests surface-level sentence similarity. It doesn't test:
- Compositional understanding ("the dog chased the cat" vs "the cat chased the dog")
- Entailment direction (asymmetric — SGS's transmittance is explicitly asymmetric)
- Negation/quantification
- Long-range dependencies

SGS's structural advantages (ordered composition, transmittance-based occlusion, explicit Gaussian primitives) should matter MORE on tasks that require structural understanding. STS-B is the WORST benchmark for SGS because it primarily tests word overlap and semantic proximity — which mean-pooling already handles well.

**Required resolution:** Test on tasks where composition structure matters: NLI classification (3-way), paraphrase detection (MRPC/QQP), and especially SCAN/COGS (compositional generalization).

---

## MODERATE CHALLENGE

### C7: Mean-Pool Got WORSE with NLI — This Is Important and Under-Analyzed

**The data:**
- Mean-pool on STS-B: 0.616
- Mean-pool on NLI: 0.605

Mean-pooling DEGRADED with contrastive training. This means the contrastive loss actively hurts unstructured representations. It pushes embeddings to be globally calibrated in a way that destroys the local structure mean-pooling relies on.

This confirms that **structured composition is necessary for contrastive learning to work.** Both SGS and softmax benefit; mean-pool does not. SGS's advantage over mean-pool WIDENED with NLI (+0.059 → +0.109). The rendering equation is validated as a necessary structural component — the question is only whether it's better than softmax's structure.

---

## Alternative Interpretations

### Alt-1: SGS Is an Initialization Method, Not an Architecture

The strongest consistent finding: SGS has the best zero-shot performance (0.689 vs 0.609). After training, the gap narrows or reverses. This suggests the VALUE of the rendering equation is as an **initialization/inductive bias**, not as a runtime composition mechanism. The optimal architecture might be: initialize with SGS, then switch to softmax for training.

### Alt-2: The Contrastive Loss Is the Wrong Test for SGS

Contrastive learning requires globally comparable embeddings. SGS's local composition is not designed for global comparability — it's designed for compositional structure. Testing SGS with contrastive loss is like testing a race car on a muddy track. Test on compositional tasks (SCAN/COGS) where structural inductive bias should dominate.

### Alt-3: SGS + Softmax Hybrid Is the Answer

The data suggests: SGS for the first pass (structural inductive bias), softmax for the second pass (flexible capacity). This combines SGS's superior initialization with softmax's superior scaling.

---

## Verdict

The Phase 2 results are more nuanced than presented. The "interesting trade-off" framing avoids the fact that the pre-defined kill condition (SGS ≤ matched softmax) was triggered on the NLI experiment. However, the kill condition was defined for STS-B training, not NLI contrastive training — so the confound (C2, C3) makes the trigger ambiguous.

**The rendering equation is NOT dead.** It clearly provides value:
- Best zero-shot performance (consistent, large margin)
- Better than softmax with limited data (Phase 1.5, significant)
- Necessary for contrastive training to work (mean-pool fails)
- Provably more expressive than softmax (Lean verified)

**But it is NOT the dominant composition mechanism at scale.** Softmax catches up with data.

**The right next step is NOT more STS-B tuning.** It's testing on tasks where the structural advantage should be decisive:
1. SCAN/COGS compositional generalization
2. NLI classification (not contrastive — 3-way prediction)
3. SGS+softmax hybrid

If SGS wins on SCAN/COGS, the paper is: "Rendering provides structural inductive bias for compositional language understanding, superior to attention on compositional tasks while attention scales better on distributional similarity."

If SGS doesn't win on SCAN/COGS, the paper is: "Rendering provides the strongest zero-shot composition mechanism for language (proven), but attention's capacity dominates at all scales for all tasks."

Both are publishable. The second is a well-characterized negative result.
