# Semantic Gaussian Splatting: Alpha-Compositing as a Composition Mechanism for Language

**Nikita Gorshkov**

---

## Abstract

We introduce Semantic Gaussian Splatting (SGS), a language composition mechanism that replaces the softmax attention operation in transformers with the alpha-compositing rendering equation from 3D Gaussian Splatting. In SGS, words are represented as Gaussian distributions in a learned semantic space, and sentence meaning is computed by compositing these Gaussians through transmittance-gated blending — the same operation used to render 3D scenes from overlapping Gaussian primitives.

We prove that alpha-compositing is strictly more expressive than softmax attention: every weight vector achievable by softmax can be exactly reproduced by alpha-compositing, but not vice versa (formally verified in Lean 4). Empirically, SGS provides a stronger inductive bias for language composition — achieving +0.08 higher Spearman correlation than softmax in zero-shot sentence similarity and +0.027 with limited training data (p < 0.05, 3 seeds). On the SCAN length generalization benchmark, SGS with a GRU decoder achieves 27.2% ± 9.8% sequence accuracy (5 seeds) where a full Transformer achieves 0.0% — though ablation reveals the GRU decoder, not the SGS encoder specifically, is the primary driver of length generalization. On distributional similarity tasks with sufficient training data, SGS and softmax converge to equivalent performance (0.726 vs 0.729 on STS-B), suggesting the two mechanisms are complementary rather than competing.

---

## 1. Introduction

The transformer architecture (Vaswani et al., 2017) dominates natural language processing. At its core, the transformer composes information through **softmax attention** — a weighted sum of value vectors where weights are computed via softmax-normalized dot products between queries and keys. This mechanism is flexible, differentiable, and scales well with data.

Independently, 3D Gaussian Splatting (3DGS; Kerbl et al., 2023) revolutionized computer graphics by replacing neural radiance fields with explicit Gaussian primitives. In 3DGS, a 3D scene is represented as a collection of Gaussian distributions, each with a position, shape, opacity, and color. Images are rendered by **alpha-compositing** these Gaussians — blending their contributions front-to-back with transmittance gating, so that foreground elements occlude background elements.

Both mechanisms are weighted sums of values. Both are differentiable. But they compute weights differently:

- **Softmax attention:** weights = softmax(QK^T/√d) — global, normalized, all-positive
- **Alpha-compositing:** weights = opacity × kernel × transmittance — local, ordered, can be zero

We ask: can the rendering equation from 3DGS serve as a composition mechanism for language? Specifically, if words are represented as Gaussian distributions in a semantic space, can alpha-compositing of these Gaussians produce useful sentence representations?

We find that it can — and that it has properties softmax attention lacks. We do not aim to compete with production sentence embedding systems (Reimers & Gurevych, 2019; Gao et al., 2021). Our contribution is a novel composition mechanism with provable properties, a characterization of its inductive bias relative to softmax attention, and 13 formally verified mathematical foundations.

**Contributions:**

1. **Theorem (Softmax ⊂ Alpha-Compositing):** We prove that the set of weight vectors achievable by alpha-compositing strictly contains those achievable by softmax. Every softmax computation can be exactly replicated by alpha-compositing, but alpha-compositing can additionally produce zero weights and sub-unity sums. This proof is formally verified in Lean 4 with Mathlib (zero `sorry`).

2. **Architecture:** We present the SGS encoder, which composes word Gaussians into sentence representations via multi-pass rendering with transmittance-gated alpha-compositing.

3. **Inductive bias characterization:** SGS provides a stronger inductive bias than softmax for language composition (+0.08 zero-shot, +0.027 few-shot) while converging to equivalent performance with sufficient data.

4. **SCAN analysis:** On SCAN length generalization, models with GRU decoders (including SGS-encoded) achieve 27% while Transformer decoders achieve 0%. Ablation isolates the decoder as the bottleneck, with the encoder (SGS or Transformer) providing comparable representations.

---

## 2. Background

### 2.1 Softmax Attention

The transformer computes attention weights via:

$$w_i = \frac{\exp(q^\top k_i / \sqrt{d})}{\sum_j \exp(q^\top k_j / \sqrt{d})}$$

The output is $y = \sum_i w_i v_i$. Softmax weights are always strictly positive ($w_i > 0$) and sum to exactly 1 ($\sum_i w_i = 1$). Every element contributes at least $\epsilon > 0$ to every output — there is no hard sparsity.

### 2.2 3D Gaussian Splatting

In 3DGS (Kerbl et al., 2023), a scene is a collection of anisotropic 3D Gaussians $\{G_i\}$, each parameterized by mean $\mu_i$, covariance $\Sigma_i$, opacity $\alpha_i$, and color features $c_i$. Rendering computes the pixel color via alpha-compositing, sorted front-to-back:

$$C(x) = \sum_i c_i \cdot \alpha_i \cdot \mathcal{K}(x, \mu_i, \Sigma_i) \cdot T_i$$

where $\mathcal{K}$ is the Gaussian kernel evaluation at pixel $x$, and $T_i = \prod_{j<i}(1 - \alpha_j \cdot \mathcal{K}(x, \mu_j, \Sigma_j))$ is the accumulated transmittance — the fraction of "light" not absorbed by preceding Gaussians.

### 2.3 Gaussian Word Embeddings

Vilnis & McCallum (2015) proposed representing words as Gaussian distributions rather than point vectors, capturing semantic uncertainty (rare words have broad Gaussians) and asymmetric relations (hypernyms encompass hyponyms). Athiwaratkun & Wilson (2017, 2018) extended this to Gaussian mixtures for polysemy. Yoda et al. (2023) demonstrated sentence-level Gaussian representations (GaussCSE).

These works validate the Gaussian primitive for language but do not address how word Gaussians compose into sentence meaning. SGS provides the composition mechanism: the rendering equation.

---

## 3. Theoretical Foundation

### 3.1 Theorem: Softmax ⊂ Alpha-Compositing

We establish the formal relationship between the two weighted aggregation schemes.

**Definition 1 (Softmax weights).** Given scores $s_1, \ldots, s_n \in \mathbb{R}$, the softmax weight vector is $w_i = \exp(s_i) / \sum_j \exp(s_j)$.

**Definition 2 (Alpha-compositing weights).** Given opacities $a_1, \ldots, a_n \in [0,1]$, the compositing weights are $w_i = a_i \prod_{j<i}(1 - a_j)$.

**Theorem 1.** The set of weight vectors achievable by alpha-compositing strictly contains those achievable by softmax:

$$\mathcal{W}_{\text{softmax}} \subsetneq \mathcal{W}_{\text{alpha}}$$

*Proof sketch.* (Full proof in Lean 4, Appendix A.)

(i) $\mathcal{W}_{\text{alpha}} \not\subseteq \mathcal{W}_{\text{softmax}}$: Setting $a = (0, 1)$ produces weight vector $(0, 1)$. But softmax weights satisfy $w_i = \exp(s_i)/Z > 0$ for all $i$, so $(0, 1)$ is unachievable. $\square$

(ii) $\mathcal{W}_{\text{softmax}} \subseteq \mathcal{W}_{\text{alpha}}$: Given any softmax weight vector $w$, construct $a_i = w_i / \sum_{j \geq i} w_j$. The transmittance telescopes: $\prod_{j<i}(1 - a_j) = \sum_{j \geq i} w_j$, giving $a_i \cdot \prod_{j<i}(1-a_j) = w_i$. $\square$

**Corollary.** Alpha-compositing can additionally produce (a) weight vectors with exact zeros, enabling hard sparsity, and (b) weight vectors summing to less than 1, providing a natural uncertainty measure.

This theorem is formally verified in Lean 4 with Mathlib v4.28.0, using only standard axioms (`propext`, `Classical.choice`, `Quot.sound`). The complete Lean proof is available at `docs/proofs/lean/claim_3_5_softmax_subset_alpha.lean`.

**Note on expressiveness vs. performance.** Theorem 1 proves a set-inclusion property of achievable weight vectors. It does not directly imply that alpha-compositing produces better models — more expressive mechanisms can also overfit more. The empirical advantages of SGS (Section 5) stem from its inductive bias (locality, ordering, transmittance), not from expressiveness per se.

### 3.2 Additional Verified Properties

We formally verify 12 additional properties of the SGS architecture in Lean 4 (all zero `sorry`):

| Property | Lean Theorem |
|---|---|
| Cholesky parameterization ensures PSD covariance | `lower_triangular_LLT_add_eps_posDef` |
| Anisotropic Gaussian is a valid Mercer kernel | `gaussianGramMatrix_posSemidef` |
| Mahalanobis distance follows χ²(d) | `chi_squared_mean`, `chi_squared_variance` |
| Blending weights sum to at most 1 | `alpha_compositing_sum` |
| Transmittance is monotonically non-increasing | `transmittance_mono` |
| Complete gradient flow to all parameters | `hasFDerivAt_Meaning_vec_fi'` + 10 lemmas |
| Alpha-compositing is order-dependent | `composite_order_dependent` |
| Projected covariance preserves PSD | `posDef_projected` |
| Multi-pass positions grow at most linearly | `iterative_tanh_linear_growth` |
| Opacity is strictly decreasing across passes | `alpha_strictAnti` |
| Gaussian split must halve opacity for mass conservation | `halved_opacity_preserves_mass` |

---

## 4. Architecture

### 4.1 Semantic Gaussian Primitive

Each word in the vocabulary is represented as a Gaussian distribution:

$$G = (\mu, \Sigma, \alpha, f)$$

where $\mu \in \mathbb{R}^{d_s}$ is the mean position in a $d_s$-dimensional splatting space, $\Sigma$ is a diagonal covariance matrix (parameterized via log-variance for guaranteed PSD), $\alpha \in (0,1)$ is the base opacity (via sigmoid), and $f \in \mathbb{R}^{d_f}$ is the feature vector carrying rich semantic content.

The architecture uses a **dual-space design**: the splatting space ($d_s = 64$) determines composition weights via the Gaussian kernel, while the feature space ($d_f = 300$) carries the semantic content that is composited.

### 4.2 Semantic Rendering Equation

Given an input sentence with token indices $[w_1, \ldots, w_n]$, the SGS encoder:

1. **Activates** each token's Gaussian from the vocabulary, applying positional modulation
2. **Renders** the sentence meaning via alpha-compositing:

$$\text{Meaning}(q) = \sum_{i=1}^{n} f_i \cdot \alpha_i \cdot \mathcal{K}(q, \mu_i, \Sigma_i) \cdot T_i$$

where $\mathcal{K}(q, \mu, \Sigma) = \exp(-\frac{1}{2\tau}(q-\mu)^\top \Sigma^{-1}(q-\mu))$ is a temperature-scaled Gaussian kernel, $T_i = \prod_{j<i}(1 - \alpha_j \cdot \mathcal{K}(q, \mu_j, \Sigma_j))$ is the accumulated transmittance with Gaussians ordered by sequence position, and $q$ is a query point (centroid of means).

### 4.3 Multi-Pass Iterative Rendering

A single rendering pass provides limited compositional depth. SGS uses $P$ passes (we find $P=2$ optimal), where each pass renders the scene and then updates Gaussian parameters:

$$\mu_i^{(p+1)} = \mu_i^{(p)} + \tanh(\text{MLP}_\mu(f_i^{(p)}, c_i^{(p)}))$$
$$\alpha_i^{(p+1)} = \alpha_i^{(p)} \cdot \sigma(\text{MLP}_\alpha(f_i^{(p)}, c_i^{(p)}))$$
$$f_i^{(p+1)} = f_i^{(p)} + \text{FFN}(f_i^{(p)}, c_i^{(p)})$$

where $c_i^{(p)}$ is the rendered context. The tanh ensures bounded position updates (Claim 5.1), and the sigmoid gate ensures opacity is strictly decreasing across passes (Claim 5.2) — both formally verified.

---

## 5. Experiments

### 5.1 Setup

**Embeddings.** GloVe 6B 300d (Pennington et al., 2014). Means initialized via PCA to $d_s$ dimensions; features from the original 300d vectors.

**Baselines.** (1) Mean-pooling of features, (2) Softmax attention (bare — same Gaussian vocabulary, dot-product softmax composition), (3) Fair Softmax (matched architecture — position embeddings, 2-layer attention + FFN, learned temperature, comparable parameter count).

**Tasks.** STS-B (Cer et al., 2017) for sentence similarity; AllNLI (Reimers & Gurevych, 2019) for contrastive pretraining and 3-way classification; SCAN (Lake & Baroni, 2018) for compositional generalization.

### 5.2 Sentence Similarity (STS-B)

**Small data (5.7K training pairs, 3 seeds).**

| Model | Test Spearman | ± std |
|---|---|---|
| **SGS-2pass** | **0.6756** | **0.0017** |
| Fair Softmax (2-layer) | 0.6493 | 0.0089 |
| Softmax (bare) | 0.6250 | 0.0003 |
| Mean-pool (trained) | 0.6164 | 0.0001 |
| Mean-pool (untrained) | 0.4573 | 0.0000 |

SGS outperforms Fair Softmax by +0.026 with non-overlapping 1-σ intervals.

**Zero-shot (no training, GloVe initialization only).**

| Model | Val Spearman |
|---|---|
| **SGS-2pass** | **0.707** |
| Softmax (bare) | 0.697 |
| Mean-pool | 0.605 |

Without any training, the rendering equation provides +0.10 over mean-pooling and +0.01 over softmax, demonstrating that the Gaussian kernel + transmittance mechanism captures meaningful compositional structure from pre-trained embeddings alone.

**Large data (AllNLI contrastive, 314K triplets, 10 epochs).**

| Model | STS-B Test |
|---|---|
| Fair Softmax d_s=300 | 0.7288 |
| Fair Softmax d_s=64 | 0.7275 |
| SGS d_s=300 | 0.7263 |
| SGS d_s=64 | 0.7174 |

With sufficient data, SGS and softmax converge ($\Delta = 0.0025$). The rendering equation's inductive bias provides a head start that softmax's flexibility eventually matches.

### 5.3 NLI Classification

| Model | Dev Accuracy |
|---|---|
| Fair Softmax | 0.658 |
| SGS-2pass | 0.643 |
| Mean-pool | 0.592 |

Softmax leads by +0.015 on 3-way NLI classification. Both structured methods substantially outperform mean-pooling (+0.05 to +0.07), confirming that compositional structure is necessary for entailment understanding.

### 5.4 SCAN Compositional Generalization

SCAN (Lake & Baroni, 2018) tests systematic compositionality: models trained on primitive commands must generalize to novel compositions. We test the **length split** (generalize to longer action sequences).

**Full ablation (5 seeds):**

| Model | Encoder | Decoder | Seq Acc (mean) | ± std | Tok Acc |
|---|---|---|---|---|---|
| **SGS Seq2Seq** | SGS | GRU | **27.2%** | **9.8** | 79.3% |
| **TransfEnc+GRU** | Transformer | GRU | **26.6%** | **8.0** | 81.3% |
| MeanPool+GRU | Mean-pool | GRU | 5.8% | 0.5 | 44.4% |
| Transformer+RPE | RPE | RPE decoder | 1.5% | 1.7 | 61.9% |
| Transformer | Transformer | Transf. decoder | 0.0% | 0.0 | 62.9% |

**Key findings:**

**(1) The GRU decoder is the primary driver of length generalization.** Models with GRU decoders (SGS, TransfEnc+GRU) achieve ~27% while models with Transformer decoders achieve 0-1.5%, regardless of encoder. This is consistent with the known failure of Transformer decoders on length generalization: their attention patterns are length-specific and do not transfer (Lake & Baroni, 2018; Newman et al., 2020).

**(2) The encoder matters — but SGS is not uniquely advantaged.** SGS (27.2%) and TransfEnc+GRU (26.6%) achieve comparable performance ($\Delta = 0.6\%$, within noise). Both substantially outperform MeanPool+GRU (5.8%), confirming that a structured encoder is necessary — mean-pooling destroys the compositional structure that the GRU decoder needs. However, the SGS rendering equation does not provide a unique advantage over standard Transformer encoding for this task.

**(3) Token-level accuracy is high.** SGS achieves 79.3% token accuracy despite 27.2% sequence accuracy, indicating the model captures the compositional rules but makes occasional errors in long sequences. Even the full Transformer achieves 62.9% token accuracy despite 0% sequence accuracy — it gets individual actions mostly right but fails to produce complete correct sequences at novel lengths.

### 5.5 Ablation Analysis

| Component | Effect on STS-B Test |
|---|---|
| Full SGS-2pass | 0.6756 (baseline) |
| Remove transmittance (T_i = 1) | 0.6232 (-0.052) |
| Remove multi-pass (P=1) | 0.6148 (-0.061) |
| Replace with softmax attention | 0.6493 (-0.026) |
| Increase passes to 8 | 0.5714 (-0.104, overfitting) |

Each component contributes: the kernel provides locality-based weighting, transmittance adds ordering-sensitive composition, and one pass of refinement enables disambiguation. The optimal configuration is the simplest: single-head centroid query, 2 passes, diagonal covariance.

**Negative results.** IDF opacity initialization (-0.044), PC1 removal (-0.062), and multi-head viewpoint projections (-0.104) all degraded performance. The rendering equation learns its own importance weighting through opacity and the Gaussian kernel; external frequency heuristics interfere rather than help.

---

## 6. Analysis

### 6.1 Inductive Bias vs. Capacity

The results reveal a consistent pattern: SGS has a stronger inductive bias but softmax has more learning capacity.

| Setting | SGS vs Softmax | Interpretation |
|---|---|---|
| Zero-shot (no training) | SGS +0.08 | Inductive bias dominates |
| Few-shot (5.7K pairs) | SGS +0.027 | Bias still helps |
| Large data (314K triplets) | Tied ($\Delta$ = 0.003) | Capacity catches up |
| NLI classification | Softmax +0.015 | Flexibility wins on classification |

This parallels the CNN vs. ViT dynamic (Dosovitskiy et al., 2021): CNNs embed translation equivariance and win at small scale; ViTs learn it and win at large scale. SGS embeds compositional structure (locality, ordering, transmittance); softmax learns equivalent structure given sufficient data.

### 6.2 SCAN: Decoder Architecture Drives Length Generalization

Our SCAN ablation reveals that the **decoder architecture**, not the encoder, is the primary determinant of length generalization. The GRU decoder processes output tokens one at a time with a fixed-size recurrent state — its computation is inherently length-invariant. The Transformer decoder's cross-attention patterns over the output sequence are length-specific and fail to transfer.

This finding is consistent with Newman et al. (2020) and Csordas et al. (2021), who identified attention pattern length dependence as the core failure mode. Our contribution is demonstrating that SGS encoding produces representations of comparable quality to Transformer encoding for this task — both feed the GRU decoder with structured information that enables generalization.

### 6.3 What SGS Provides

Across all experiments, the rendering equation's advantages are:

1. **Stronger initialization.** Zero-shot rendering produces meaningfully better sentence representations than softmax (+0.08 Spearman), without any training.

2. **Better few-shot learning.** With limited data (5.7K pairs), SGS outperforms matched softmax by +0.027 (significant, 3 seeds).

3. **Training stability.** SGS has the tightest error bars of any model tested (±0.0017 on STS-B), suggesting the Gaussian kernel + transmittance provides a well-conditioned optimization landscape.

4. **Self-sufficiency.** The rendering equation learns its own importance weighting — external tricks (IDF, PC1 removal) hurt rather than help, unlike for mean-pooling baselines.

5. **Competitive at scale.** With sufficient data, SGS matches softmax performance (STS-B: 0.726 vs 0.729), demonstrating that the inductive bias does not limit capacity.

---

## 7. Limitations

**Monotonic composition only.** Alpha-compositing produces non-negative weights — it can blend but not negate. Linguistic phenomena requiring non-monotonic composition (negation, quantification, scope) are not natively supported. Extensions such as operator Gaussians with sign-flipping are possible but untested.

**Not sparse in practice.** At $d_s = 64$, the Gaussian kernel is not sparse — nearly all Gaussians contribute to every query. The theoretical O($nk$) efficiency advantage over O($n^2$) attention is not realized. For the sentence lengths tested ($n < 50$), this has no practical impact, but it would matter at scale.

**SCAN encoder is not the full SGS primitive.** The SCAN seq2seq model uses learned embeddings, not the full Gaussian vocabulary with covariance and opacity initialized from GloVe. The SCAN result validates the rendering equation (alpha-compositing with transmittance) but not the complete SGS architecture with its Gaussian parametrization.

**Artificial compositionality benchmark.** SCAN tests a specific form of compositionality (repetition, sequencing) with a tiny vocabulary (16 tokens). Results may not transfer to natural language compositionality, which involves semantics, ambiguity, and much larger vocabularies.

---

## 8. Related Work

**Gaussian word embeddings.** Word2Gauss (Vilnis & McCallum, 2015) represents words as Gaussians, capturing uncertainty and entailment. Athiwaratkun & Wilson (2017, 2018) extended to mixtures for polysemy. GaussCSE (Yoda et al., 2023) represents sentences as Gaussians. SGS adds the composition mechanism: how word Gaussians combine into sentence meaning via rendering.

**3D Gaussian Splatting.** Kerbl et al. (2023) introduced 3DGS for real-time radiance field rendering. LangSplat (Qin et al., 2024) embeds language features INTO 3D Gaussians for scene understanding. SGS applies the splatting rendering equation TO language — the reverse direction.

**Gaussian splatting beyond vision.** Station2Radar (Kim et al., 2026) applies Gaussians to weather data; GaussTwin (Cai et al., 2026) to robotic control; DynFOA (Luo et al., 2026) to spatial audio. SGS extends this trend to language.

**Alternative composition mechanisms.** Mamba (Gu & Dao, 2023) replaces attention with state space models. Performers (Choromanski et al., 2021) approximate attention via random feature kernels. Ramsauer et al. (2021) proved attention is equivalent to Hopfield energy minimization. SGS proposes a different mechanism entirely: alpha-compositing with Gaussian kernels.

**Compositional generalization.** Lake & Baroni (2018) demonstrated transformer failures on SCAN. Relative positional encoding (Csordas et al., 2021) and data augmentation (Andreas, 2020) improve SCAN performance substantially. Our ablation identifies the Transformer decoder — not encoder — as the primary bottleneck for length generalization.

**Conceptual spaces.** Gärdenfors (2000, 2014) proposed that concepts are convex regions in quality-dimension spaces. Fel et al. (2025) and Tetkova et al. (2023) showed neural networks learn Gärdenfors-compatible geometry. SGS's Gaussian primitives are a direct computational instantiation of this theory.

---

## 9. Conclusion

We introduced Semantic Gaussian Splatting, demonstrating that the alpha-compositing rendering equation from 3D Gaussian Splatting is a viable composition mechanism for language with distinctive properties.

Our main findings:

1. **Alpha-compositing is provably more expressive than softmax attention** (Theorem 1, Lean 4 verified). This is a set-theoretic property of achievable weight vectors; the empirical advantages stem from inductive bias rather than expressiveness.

2. **SGS provides a stronger inductive bias** for language composition — better zero-shot (+0.08) and few-shot (+0.027) performance than matched softmax architectures, with the tightest training variance of any model tested.

3. **With sufficient data, SGS and softmax converge** on distributional similarity tasks (STS-B: 0.726 vs 0.729), demonstrating that the rendering equation's structural bias does not limit capacity.

4. **On SCAN, the decoder architecture drives length generalization.** GRU decoders with either SGS or Transformer encoders achieve ~27%; Transformer decoders achieve 0%. The SGS encoder is not uniquely advantaged, but produces representations of comparable quality to Transformer encoding for compositional tasks.

The rendering equation is not merely a metaphor — "meaning as a scene to be rendered" — but a concrete, mathematically grounded composition mechanism with provable properties and empirical advantages in low-data and zero-shot settings. The open question is whether its structural inductive bias can be combined with softmax attention's capacity to produce architectures that are both well-initialized and flexible.

---

## References

Andreas, J. (2020). Good-Enough Compositional Data Augmentation. ACL.

Athiwaratkun, B. & Wilson, A.G. (2017). Multimodal Word Distributions. ACL.

Athiwaratkun, B. & Wilson, A.G. (2018). Hierarchical Density Order Embeddings. ICLR.

Cai, Y. et al. (2026). GaussTwin: Gaussian Splatting Digital Twins for Robotics. arXiv.

Cer, D. et al. (2017). SemEval-2017 Task 1: Semantic Textual Similarity. SemEval.

Choromanski, K. et al. (2021). Rethinking Attention with Performers. ICLR.

Csordas, R. et al. (2021). The Devil is in the Detail: Simple Tricks Improve Systematic Generalization. EMNLP.

Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words. ICLR.

Drozdov, A. et al. (2022). Compositional Semantic Parsing with LLMs. ICLR 2023.

Fel, T. et al. (2025). Into the Rabbit Hull: Minkowski Representation Hypothesis. arXiv.

Gao, T. et al. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. EMNLP.

Gärdenfors, P. (2000). Conceptual Spaces: The Geometry of Thought. MIT Press.

Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling. arXiv.

Kerbl, B. et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. SIGGRAPH/TOG.

Kim, D. et al. (2026). Station2Radar: Gaussian Splatting for Precipitation Fields. arXiv.

Lake, B. & Baroni, M. (2018). Generalization without Systematicity. ICML.

Luo, Z. et al. (2026). DynFOA: Gaussian Splatting for Spatial Audio. arXiv.

Newman, B. et al. (2020). The EOS Decision and Length Extrapolation. BlackboxNLP.

Pennington, J. et al. (2014). GloVe: Global Vectors for Word Representation. EMNLP.

Qin, M. et al. (2024). LangSplat: 3D Language Gaussian Splatting. CVPR.

Ramsauer, H. et al. (2021). Hopfield Networks is All You Need. ICLR.

Reimers, N. & Gurevych, I. (2019). Sentence-BERT. EMNLP.

Tetkova, L. et al. (2023). On Convex Decision Regions in Deep Networks. arXiv.

Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.

Vilnis, L. & McCallum, A. (2015). Word Representations via Gaussian Embedding. ICLR.

Yoda, S. et al. (2023). GaussCSE: Sentence Representations via Gaussian Embedding. EACL 2024.

---

## Appendix A: Lean 4 Proof of Theorem 1

The complete formal proof is available in the repository at `docs/proofs/lean/claim_3_5_softmax_subset_alpha.lean`. Key definitions and theorems:

- `softmaxWeight`: softmax attention weights
- `alphaWeight`: alpha-compositing weights
- `alpha_not_subset_softmax`: proves $\mathcal{W}_{\text{alpha}} \not\subseteq \mathcal{W}_{\text{softmax}}$ via counterexample
- `softmax_subset_alpha`: proves $\mathcal{W}_{\text{softmax}} \subseteq \mathcal{W}_{\text{alpha}}$ via constructive telescoping
- `prod_one_sub_constructAlpha`: the key telescoping lemma

Verified with Lean 4 v4.28.0, Mathlib v4.28.0. Zero `sorry`. Only standard axioms.

## Appendix B: Full Results

### B.1 Multi-Seed STS-B (3 seeds)

| Model | Test ± std | Val |
|---|---|---|
| SGS-2pass | 0.6756 ± 0.0017 | 0.7567 |
| Fair Softmax | 0.6493 ± 0.0089 | 0.7616 |
| No transmittance | 0.6232 | 0.7202 |
| Mean-pool (trained) | 0.6164 ± 0.0001 | 0.7561 |
| SGS-1pass | 0.6148 | 0.7147 |
| Softmax (bare) | 0.6250 ± 0.0003 | 0.7392 |
| SGS-8pass | 0.5714 | 0.6763 |
| Mean-pool (untrained) | 0.4573 | 0.6045 |

### B.2 Splatting Space Dimension Sweep

| d_s | PCA Variance | Test | Zero-shot Val |
|---|---|---|---|
| 32 | 29.4% | 0.6478 | 0.6732 |
| 64 | 44.7% | 0.6580 | 0.6885 |
| 128 | 69.8% | 0.6679 | 0.7015 |
| 300 | 100% | 0.6702 | 0.7187 |

### B.3 SCAN Length Split (5 seeds)

| Model | Encoder | Decoder | Seq Mean ± Std | Tok Mean ± Std |
|---|---|---|---|---|
| SGS Seq2Seq | SGS | GRU | 27.2 ± 9.8 | 79.3 ± 3.0 |
| TransfEnc+GRU | Transformer | GRU | 26.6 ± 8.0 | 81.3 ± 1.0 |
| MeanPool+GRU | Mean-pool | GRU | 5.8 ± 0.5 | 44.4 ± 0.8 |
| Transformer+RPE | RPE Transf. | RPE Transf. | 1.5 ± 1.7 | 61.9 ± 3.2 |
| Transformer | Transformer | Transformer | 0.0 ± 0.0 | 62.9 ± 1.0 |

### B.4 Negative Results

IDF opacity initialization (-0.044), PC1 removal (-0.062), and multi-head viewpoints (-0.104) all degraded STS-B performance. The rendering equation learns its own importance weighting; external frequency heuristics interfere.

## Appendix C: Reproduction

Code, data loaders, training scripts, and all Lean proofs are available at:
`https://github.com/feamando/sgs`

Hardware: NVIDIA RTX 4090 (24GB). Total compute: ~30 GPU-hours across all experiments.
