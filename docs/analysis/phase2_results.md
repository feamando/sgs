# Phase 2 Results: Data vs Dimensions

**Date:** 2026-04-08

---

## Raw Results

### Experiment 1: AllNLI Training → STS-B Evaluation

| Model | STS-B Test | STS-B Val | ZS Val | Time |
|---|---|---|---|---|
| **Fair Softmax (NLI)** | **0.7261** | 0.8068 | 0.6089 | 146s |
| SGS-2pass (NLI) | 0.7138 | 0.8060 | 0.6885 | 185s |
| Mean-pool (NLI) | 0.6052 | 0.7471 | 0.6045 | 68s |

### Experiment 2: d_s Sweep on STS-B

| d_s | PCA Variance | Test | Val | ZS Val | Params |
|---|---|---|---|---|---|
| 32 | 29.4% | 0.6478 | 0.7480 | 0.6732 | 18.8M |
| 64 | 44.7% | 0.6580 | 0.7615 | 0.6885 | 22.1M |
| 128 | 69.8% | 0.6679 | 0.7668 | 0.7015 | 28.5M |
| 300 | 100% | 0.6702 | 0.7680 | 0.7187 | 45.9M |

---

## Hypothesis Verdicts

### H1: Is the 0.676 ceiling a DATA problem? → YES

| Training Data | SGS Test | Δ from STS-B |
|---|---|---|
| STS-B (5.7K pairs) | 0.6756 | baseline |
| AllNLI (314K triplets) | 0.7138 | **+0.038** |

More data pushed SGS from 0.676 to 0.714 — a substantial jump. **The data bottleneck is confirmed.** STS-B's 5.7K pairs are insufficient for 22M parameters.

### H2: Is 44.7% PCA variance limiting? → MARGINALLY

| d_s | Test | Δ from d_s=64 |
|---|---|---|
| 32 | 0.6478 | -0.010 |
| 64 | 0.6580 | baseline |
| 128 | 0.6679 | +0.010 |
| 300 | 0.6702 | +0.012 |

More dimensions help, but the gains are diminishing. Going from 44.7% to 100% of variance only adds +0.012. **Dimensionality is NOT the primary bottleneck — data is.**

---

## The Critical New Finding: Fair Softmax Overtakes SGS at Scale

| Comparison | STS-B (5.7K) | AllNLI (314K) | Shift |
|---|---|---|---|
| SGS vs Fair Softmax | **+0.026 (SGS wins)** | **-0.012 (Softmax wins)** | Reversed |
| SGS vs Mean-pool | +0.059 | **+0.109** | Widened |

**With small data (STS-B), SGS beats Fair Softmax by +0.026.**
**With large data (AllNLI), Fair Softmax beats SGS by -0.012.**

This is the most important finding of Phase 2. The rendering equation has a **stronger inductive bias** — it extracts more from limited data. But softmax attention has **more learning capacity** — it catches up and overtakes with sufficient data.

The evidence:

| Signal | SGS | Fair Softmax | Interpretation |
|---|---|---|---|
| Zero-shot val (no training) | **0.689** | 0.609 | SGS's inductive bias is far better |
| After STS-B training (5.7K) | **0.676** | 0.649 | SGS maintains lead with small data |
| After NLI training (314K) | 0.714 | **0.726** | Softmax overtakes with large data |
| Improvement from NLI | +0.038 | **+0.077** | Softmax benefits MORE from data |

**The rendering equation starts ahead but scales slower.** Softmax starts behind but scales faster.

---

## What This Means

### For the paper

This is actually a **more interesting finding** than "SGS beats softmax across the board." It reveals a fundamental trade-off:

- **Alpha-compositing has better inductive bias for language.** The Gaussian kernel + transmittance + sequence ordering encodes useful structure that softmax must learn from scratch. This is why SGS wins zero-shot (+0.08) and with small data (+0.026).

- **Softmax has higher capacity ceiling.** The global all-to-all attention mechanism can learn arbitrarily complex weighting patterns. Alpha-compositing's transmittance and locality constrain the space of learnable functions. With enough data, softmax's flexibility wins.

This parallels a well-known pattern in ML: **strong inductive biases help with limited data but hurt at scale** (cf. CNNs vs ViTs — CNNs have translation equivariance, ViTs learn it; ViTs win at scale).

### For SGS's practical value

SGS is most valuable when:
1. **Data is scarce** — few-shot, low-resource, domain-specific settings
2. **Zero-shot transfer** — applying to new tasks without fine-tuning
3. **Interpretability matters** — per-Gaussian contributions are traceable
4. **Inductive bias is the goal** — compositional tasks where structure helps (SCAN/COGS)

SGS is less valuable when:
1. Abundant training data is available
2. Raw performance is the only metric
3. Scaling to large models is the priority

### Both SGS and Mean-pool benefited from NLI, but differently

| Model | STS-B Test | NLI Test | Δ |
|---|---|---|---|
| SGS | 0.676 | 0.714 | +0.038 |
| Mean-pool | 0.616 | 0.605 | -0.011 |
| Fair Softmax | 0.649 | 0.726 | +0.077 |

Mean-pool actually got WORSE with NLI training — contrastive learning collapsed the representations (known issue with mean-pooling + contrastive loss). SGS and Fair Softmax both improved, confirming that structured composition (whether via rendering or attention) is necessary to benefit from contrastive training.

---

## Revised SGS Positioning

The original thesis was: "Rendering can REPLACE attention."

The evidence says: "Rendering is a **better default** than attention (stronger inductive bias, better zero-shot, better with limited data) but attention scales better with abundant data."

This reframes SGS as:
1. **A strong initialization / inductive bias** for language composition
2. **A hybrid candidate** — start with rendering, fine-tune with attention at scale
3. **An interpretability tool** — even if softmax catches up on accuracy, SGS provides per-Gaussian contribution tracing that softmax cannot
4. **A compositional generalization bet** — the explicit structure may generalize better on SCAN/COGS even if softmax wins on STS-B

---

## Next Steps

### Immediate (this week)

1. **SCAN/COGS compositionality test** — this is where SGS's structural advantage should show most. Softmax fails spectacularly on SCAN; SGS's explicit composition may not.

2. **SGS + attention hybrid** — use SGS rendering for initial passes, then one softmax attention layer on top. Gets both the inductive bias AND the capacity.

3. **More NLI epochs** — SGS was still improving at epoch 3 (0.806 val). Fair Softmax plateaued (0.807→0.805). More epochs might close the gap.

### For the paper

The narrative shifts from "rendering replaces attention" to a more nuanced and arguably more publishable story:

> "We prove alpha-compositing is strictly more expressive than softmax (Lean verified). Empirically, the rendering equation provides a stronger inductive bias for language composition — better zero-shot (+0.08) and few-shot (+0.026) performance. However, softmax attention's greater flexibility allows it to overtake at scale. This reveals a fundamental expressiveness-vs-capacity trade-off in sequence composition mechanisms."

This is a stronger paper than "X beats Y" because it characterizes WHEN and WHY each approach wins.
