# SCAN Multi-Seed Results — Orthogonal Challenge

**Challenger Role:** Hostile reviewer, second round
**Date:** 2026-04-09

---

## The Results Under Review

| Model | Mean | ± Std | Min | Max |
|---|---|---|---|---|
| SGS Seq2Seq | 27.2% | 9.8% | 15.2% | 43.7% |
| Transformer+RPE | 1.5% | 1.7% | 0.0% | 4.4% |
| Transformer | 0.0% | 0.0% | 0.0% | 0.0% |

---

## CRITICAL

### C1: 27% Is Not Good — It's a 73% Failure Rate

The paper frames 27.2% as a breakthrough. A hostile reviewer reads it differently: **SGS fails on 73% of test sequences.** This is not a model you would deploy. The SCAN length split has been solved to 99%+ by other methods (data augmentation, meta-learning, specialized architectures). 27% is interesting mechanistically but not competitive.

The paper should NOT claim "SGS solves compositional generalization." It should claim "SGS demonstrates non-trivial compositional generalization from the rendering equation alone, without any compositional-generalization-specific inductions." The framing matters.

**Required fix:** Scope the claim. 27% is evidence of structural inductive bias, not a SOTA result. Explicitly compare to methods that achieve >90% (CGPS, data augmentation, meta-learning) and explain that SGS achieves this with NO compositional-specific design — it's a general composition mechanism that happens to generalize.

---

### C2: The Variance Is Enormous (±9.8 on 27.2 = 36% coefficient of variation)

The std is 36% of the mean. Seed 42 gets 43.7%; seed 1337 gets 15.2% — a 3× difference. This means the result is real but unstable. A reviewer will ask: "What determines whether SGS gets 15% or 44%? Is this random initialization luck, or is there a systematic factor?"

If the variance is driven by whether the GRU decoder happens to initialize in a good region, then it's the decoder, not the rendering equation, that determines success. The rendering equation provides signal, but the decoder's ability to exploit that signal is seed-dependent.

**Required fix:** Analyze what differs between the best seed (42, 43.7%) and worst seed (1337, 15.2%). Is it the encoder representations? The decoder initialization? The learned temperature τ? Without this analysis, the variance is unexplained noise.

---

### C3: The RPE Baseline May Be Under-Trained

The RPE transformer achieves 1.5%. The literature claims 70-90% for RPE on SCAN length. The gap is enormous. Possible reasons:

1. **Model too small:** Our RPE transformer is 936K params, d_model=128. Literature uses d_model=256-512, more layers.
2. **Too few epochs:** 20 epochs may be insufficient. Literature often trains 100+ epochs on SCAN.
3. **Learning rate:** 1e-3 with Adam may not be optimal for RPE; literature often uses lower rates with warmup.
4. **Implementation bug:** The RPE implementation is new and untested against a reference.

If the RPE baseline is broken or under-configured, the comparison is meaningless. A reviewer who knows the SCAN literature will immediately flag this: "Your RPE baseline gets 1.5% while Csordas et al. (2021) report 70%+ with RPE. Something is wrong with your baseline."

**Required fix:** Either (a) verify the RPE implementation against a reference and tune hyperparameters to match literature results, or (b) explicitly report the discrepancy and explain it's due to model size / training budget, and cite the literature numbers: "With larger models and more training, RPE-based transformers achieve 70-90% (Csordas et al., 2021). Our small-model comparison isolates the effect of the composition mechanism at matched scale."

---

## MAJOR

### C4: SGS Uses a GRU Decoder; Transformers Don't

SGS Seq2Seq: **SGS encoder + GRU decoder** (355K params)
Transformer: **Transformer encoder + Transformer decoder** (963K params)
Transformer+RPE: **RPE encoder + RPE decoder** (936K params)

The decoder is different. The GRU is sequential and inherently length-generalizing — it processes one token at a time with a recurrent state that doesn't depend on sequence length. The Transformer decoder uses attention over all previous tokens, with attention patterns that ARE length-dependent.

**The length generalization might come from the GRU decoder, not the SGS encoder.**

To test this: run "GRU decoder + mean-pool encoder" and "GRU decoder + Transformer encoder." If a GRU decoder on top of any encoder achieves ~27%, the SGS encoder contributes nothing.

**Required fix:** Add a "Transformer encoder + GRU decoder" ablation to isolate the encoder contribution.

---

### C5: Parameter Count Isn't Matched

| Model | Params |
|---|---|
| SGS Seq2Seq | 355K |
| Transformer | 963K |
| Transformer+RPE | 936K |

SGS has 2.7× fewer parameters. This could mean:
- SGS is more parameter-efficient (positive)
- The Transformer is over-parameterized for SCAN and overfits (explains 0%)
- A smaller Transformer might perform better

The parameter mismatch makes the comparison imprecise. A fairer test would match parameters.

**Required fix:** Either (a) scale SGS up to ~900K params, or (b) scale the transformers down to ~350K, or (c) acknowledge the mismatch and argue that SGS's efficiency is a feature, not a confound.

---

### C6: SCAN Is an Artificial Benchmark — Does This Transfer to Real Language?

SCAN maps commands to action sequences: "jump left twice" → "I_TURN_LEFT I_JUMP I_TURN_LEFT I_JUMP". This is:
- A tiny vocabulary (16 input, 9 output tokens)
- Perfectly deterministic (one correct answer)
- Purely syntactic (no semantics, no ambiguity)
- Not natural language

The paper claims "structural composition enables generalization that attention cannot achieve." But SCAN tests a very specific kind of compositionality (repetition, sequencing) that may not transfer to real language composition (adjective-noun modification, relative clauses, negation).

A reviewer will ask: "Does this work on COGS (Semantic parsing)? On CFQ (Complex queries)? On any task with real words?"

**Required fix:** Acknowledge SCAN's limitations explicitly. State that the result demonstrates the mechanism's potential for compositional structure, not that it solves NLP compositionality. Ideally, add one real-language compositional task (COGS would be the best candidate).

---

## MODERATE

### C7: The Training Loss Diverges From Test Accuracy in Interesting Ways

Looking at the per-seed details:

```
Seed 42:  loss 0.0037 at epoch 20, accuracy peaks at 43.7% epoch 15
Seed 123: loss 0.0032 at epoch 20, accuracy 20.3% and still rising
Seed 789: loss 0.0032 at epoch 20, accuracy 30.7% and still rising
```

The training loss converges to similar values (~0.003) across seeds, but test accuracy varies 3×. This means the model learns to fit the training data equally well regardless of seed, but generalizes very differently. The generalization is not determined by how well the model fits training — it's determined by the structure of the learned representations.

This is actually an interesting finding that the paper doesn't discuss. What makes some representations more generalizable than others? Is it the learned τ? The Gaussian positions? The GRU hidden state initialization?

**Required fix:** Note this observation in the paper. The training-test decoupling suggests the generalization comes from the representational structure, not the optimization quality.

---

### C8: No Token-Level Accuracy Reported

The paper reports sequence-level exact match (the entire output must be perfect). This is the standard SCAN metric, but it's harsh — getting 23 out of 24 tokens right counts as 0%.

Reporting token-level accuracy would show whether SGS is "almost right" on the 73% it gets wrong, or completely wrong. If token accuracy is 95% even when sequence accuracy is 27%, the model understands the compositional structure but makes occasional errors in long sequences. If token accuracy is 50%, the model is guessing.

**Required fix:** Report both sequence-level and token-level accuracy.

---

## Summary

| # | Issue | Severity | Fix |
|---|---|---|---|
| C1 | 27% is not SOTA, scope the claim | Critical | Text edit |
| C2 | Variance is 36% CV, unexplained | Critical | Analyze best vs worst seed |
| C3 | RPE baseline may be under-configured | Critical | Acknowledge gap, cite literature |
| C4 | GRU decoder may explain length generalization | Major | Add Transformer-encoder + GRU-decoder ablation |
| C5 | Parameter count unmatched (355K vs 963K) | Major | Acknowledge; argue efficiency |
| C6 | SCAN is artificial, not real language | Major | Acknowledge limitations |
| C7 | Training loss same but test accuracy 3× different | Moderate | Report this observation |
| C8 | No token-level accuracy | Moderate | Report token-level alongside sequence-level |

**The result holds, but the framing needs major adjustment.** 27.2% is not "SGS solves compositional generalization." It's "SGS demonstrates non-trivial compositional structure transfer that attention-based models completely lack, even with RPE, at matched model scale." The GRU decoder confound (C4) is the most actionable — if the encoder contribution can be isolated, the claim is much stronger.
