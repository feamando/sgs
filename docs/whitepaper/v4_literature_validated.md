# Semantic Gaussian Splatting: A Novel Architecture for Natural Language Understanding Through Radiance Field Principles

**Authors:** Nikita Gorshkov
**Date:** April 2026
**Status:** Draft — Experimental Proposal (Literature Review Complete)
**Version:** 2.0

**Companion documents:**
- `literature_review.md` — Full literature review (70+ papers)
- `v2_challenge.md` — Orthogonal adversarial challenge (12 issues)
- `v3_final.md` — Post-challenge revision

---

## 1. Executive Summary

This whitepaper proposes **Semantic Gaussian Splatting (SGS)** — a fundamentally new approach to natural language modeling that replaces the transformer's implicit, attention-based architecture with an explicit, geometric representation inspired by 3D Gaussian Splatting (3DGS). In SGS, every linguistic unit — letter, word, phrase, concept — is represented as a Gaussian distribution in a semantic space, parameterized by position (meaning), covariance (semantic breadth/uncertainty), opacity (salience), and feature vectors (attributes). Sentence understanding becomes a differentiable rendering problem: compositing overlapping semantic Gaussians from a given "viewpoint" (query/context) into a coherent output.

The core hypothesis:

> **If 3D Gaussian Splatting can replace neural radiance fields with explicit, interpretable, efficiently renderable Gaussian primitives for visual scenes, then an analogous "Semantic Gaussian Splatting" can replace transformer representations with explicit, interpretable, efficiently composable Gaussian primitives for natural language.**

### What's New in v2.0

A comprehensive literature review of 85+ papers strengthens the proposal in several ways:

1. **Gaussians for language are not speculative** — a decade of published work (2015-2026) validates Gaussian representations at word, sentence, and document levels
2. **Sentence-level Gaussians already work** — GaussCSE (EACL 2024) represents sentences as Gaussians with asymmetric similarity. SGS's contribution is the *composition mechanism* (how word Gaussians compose into sentence Gaussians via rendering)
3. **Gaussian splatting already generalizes beyond vision** — weather data (Station2Radar, 2026), robotics (GaussTwin, 2026), audio (DynFOA, 2026), brain signals (Mind-to-Face, 2025)
4. **Conceptual spaces theory is computationally validated** — LLMs learn Gardenfors-compatible convex regions (Fel et al., 2025; Kumar et al., 2025; Tetkova et al., 2023)
5. **A novel theoretical contribution identified** — no paper formally connects alpha-compositing with attention; SGS can provide this

---

## 2. Current State: How Transformers Model Language

### 2.1 The Transformer Paradigm

Since Vaswani et al. (2017), the transformer architecture has dominated NLP. Its core mechanism — scaled dot-product attention — computes:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

Tokens are embedded as point vectors. Meaning is constructed implicitly through layers of attention and feedforward transformations.

### 2.2 Known Limitations

| Limitation | Description |
|---|---|
| **Quadratic complexity** | Self-attention scales O(n^2) with sequence length. |
| **No explicit geometric structure** | Meaning geometry emerges implicitly. Power et al. (2022) showed networks can discover geometric structure but only as emergent phenomena. |
| **Compositional generalization failure** | Lake & Baroni (2018) demonstrated via SCAN that neural sequence models "fail spectacularly" at systematic compositionality. Drozdov et al. (ICLR 2023) showed explicit decomposition is needed; Anil et al. (2022) showed length generalization fails regardless of model size. |
| **Anisotropic embedding spaces** | BERT embeddings concentrate in a narrow cone (Ethayarajh, 2019; Li et al., 2020), distorting similarity measurements. |
| **Point representations** | Each token is a single point. Cannot natively represent uncertainty, semantic breadth, or graded category membership. |

### 2.3 The Alternative Architecture Landscape

SGS does not exist in a vacuum. Several non-transformer architectures have emerged, each addressing different limitations:

| Architecture | Key Idea | Complexity | SGS Relationship |
|---|---|---|---|
| **Mamba/S4** (Gu & Dao, 2023) | Selective state spaces; continuous-time dynamics | O(n) | SGS shares the continuous-field intuition but uses explicit Gaussians instead of implicit state matrices |
| **Diffusion-LM** (Li et al., 2022) | Iterative denoising in continuous embedding space | O(n·T) steps | SGS's multi-pass rendering is analogous to diffusion's iterative refinement; both operate in continuous space |
| **Performers** (Choromanski et al., 2021) | Random feature kernel approximation of attention | O(n) | SGS makes the kernel choice explicit (Gaussian) rather than approximating softmax |
| **RWKV** (Peng et al., 2023) | Linear attention with exponential decay | O(n) | RWKV's decay is a 1D radial basis function; SGS generalizes to multi-dimensional Gaussians |
| **Hopfield Networks** (Ramsauer et al., 2021) | Attention as energy minimization | O(n^2) | SGS's Gaussian scene IS the energy landscape; rendering IS energy-based retrieval |

**SGS's unique position:** It is the only proposal that combines (a) explicit geometric primitives with (b) a physics-inspired composition operator (rendering equation) and (c) adaptive density control. SSMs are continuous but implicit. Diffusion models operate in continuous space but without explicit primitives. Kernel attention replaces softmax but keeps the flat architecture. SGS provides structure at every level.

### 2.3 What Transformers Get Right

- **Depth of computation**: 12-96+ layers enable iterative refinement and in-context learning
- **Autoregressive generation**: Sequential generation respects left-to-right dependencies
- **Scalability**: Predictable scaling laws (Kaplan et al., 2020)
- **Dense training signal**: Cross-entropy on token prediction

---

## 3. How Gaussian Splatting Works: A Technical Foundation

### 3.1 Origins: From NeRF to 3DGS

Neural Radiance Fields (NeRF; Mildenhall et al., 2020) represented scenes as continuous volumetric functions. 3D Gaussian Splatting (3DGS; Kerbl et al., 2023) replaced this with explicit Gaussian primitives:

| Parameter | Symbol | Meaning |
|---|---|---|
| **Position** | μ ∈ ℝ³ | Center in 3D space |
| **Covariance** | Σ = RSS^TR^T | Shape/orientation (rotation quaternion + scale) |
| **Opacity** | α ∈ [0, 1] | Transparency |
| **Color features** | SH coefficients | View-dependent appearance |

### 3.2 The Rendering Equation

```
C = Σᵢ cᵢ αᵢ Gᵢ(x) ∏ⱼ₌₁ⁱ⁻¹ (1 − αⱼ Gⱼ(x))
```

Alpha-blending of Gaussians sorted front-to-back, fully differentiable.

### 3.3 Training: Adaptive Density Control

- **Splitting**: Large Gaussians in high-error regions → split for detail
- **Pruning**: Near-zero opacity → remove
- **Cloning**: Under-represented regions → duplicate

### 3.4 Why 3DGS Succeeded — And the Critical Caveat

3DGS demonstrated that explicit primitives beat implicit representations on speed (100-1000x), interpretability, editability, and training efficiency.

**Critical caveat for language**: 3DGS benefits from dense, pixel-level supervision. Language has sparse, token-level loss. This asymmetry remains the top risk (R1).

### 3.5 3DGS Beyond Vision — New Evidence

The literature review reveals that Gaussian splatting is rapidly expanding beyond visual reconstruction:

| Paper | Year | Domain | What Gaussians Represent |
|---|---|---|---|
| **Station2Radar** (Kim et al.) | 2026 | Meteorology | Precipitation fields from sparse weather stations |
| **GaussTwin** (Cai et al.) | 2026 | Robotics | Differentiable world models for manipulation |
| **Mind-to-Face** (Xiong et al.) | 2025 | Neuroscience | Facial expressions decoded from EEG signals |
| **DynFOA** (Luo et al.) | 2026 | Audio | Spatial audio (ambisonics) from scene geometry |
| **PIN-WM** (Li et al.) | 2025 | Physics | Rigid body dynamics world models |
| **D-REX** (Lou et al.) | 2026 | Robotics | Sim-to-real transfer for dexterous grasping |

**Significance for SGS**: The Gaussian primitive is demonstrably domain-agnostic. Weather data, robot policies, brain signals, and audio have all been successfully represented via Gaussian splatting. Language is a natural next frontier. The generality argument is no longer speculative — it's supported by a growing body of cross-domain evidence.

---

## 4. The Bridge: Why the Analogy Holds — Strengthened by Literature

### 4.1 A Decade of Gaussian Language Representations (2015-2026)

The literature review reveals that SGS is not building on a single paper (Word2Gauss) but on a sustained 11-year research trajectory:

**Word-Level Gaussians (2015-2021):**

| Paper | Year | Venue | Contribution | SGS Relevance |
|---|---|---|---|---|
| **Vilnis & McCallum** | 2015 | ICLR | Words as Gaussian distributions | Foundational primitive |
| **Athiwaratkun & Wilson** | 2017 | ACL | Gaussian *mixtures* for polysemy | Multiple splats per word |
| **Athiwaratkun & Wilson** | 2018 | ICLR | Hierarchical density order embeddings | Covariance = generality level |
| **Athiwaratkun, Wilson & Anandkumar** | 2018 | ACL | Probabilistic FastText (subword + Gaussian) | Multi-scale Gaussians |
| **Chen et al. (GMSG)** | 2015 | arXiv | Dynamic Gaussian Mixture Skip-gram with adaptive component count | **Direct precedent for adaptive density control** |
| **Yuksel et al.** | 2021 | IEEE/ACM TASLP | Gaussian covariance for semantic change detection | Temporal Gaussian dynamics |
| **Huang** | 2026 | arXiv | Gaussian Joint Embeddings for self-supervised learning | Paradigm still active in 2026 |

**Key insight from this trajectory**: Word2Gauss didn't fail and get abandoned — it spawned a productive lineage. The covariance-based representations consistently outperform points for uncertainty, entailment, and polysemy. What's missing is scaling to *composition* — exactly what SGS proposes.

**Sentence-Level Gaussians (2015-2024):**

| Paper | Year | Venue | Contribution | SGS Relevance |
|---|---|---|---|---|
| **Yoda et al. (GaussCSE)** | 2023 | EACL 2024 | **Sentences as Gaussian distributions** — asymmetric similarity, entailment direction | **Closest precedent.** Proves the output representation works. SGS adds the composition mechanism |
| **Das et al. (Gaussian LDA)** | 2015 | ACL | Topics as Gaussian regions in embedding space | "Gaussian splatting" for topics |
| **Yurochkin et al.** | 2019 | NeurIPS | Documents as distributions-of-distributions via hierarchical optimal transport | Multi-scale distributional composition |

**Assessment**: The word-to-sentence gap is the core novelty of SGS. Words as Gaussians: validated. Sentences as Gaussians: validated (GaussCSE). How word Gaussians compose into sentence Gaussians: **this is SGS's novel contribution** — the rendering equation as the composition mechanism.

### 4.2 Conceptual Spaces Theory — Computationally Validated

The orthogonal challenge questioned whether prototype theory generalizes beyond concrete nouns. The literature review provides stronger evidence:

| Paper | Year | Key Finding |
|---|---|---|
| **Fel et al.** (Minkowski Representation Hypothesis) | 2025 | Vision model tokens are "convex mixtures of archetypes" — Gardenfors-compatible |
| **Tetkova et al.** | 2023 | Neural networks form convex decision regions across images, audio, text, medical imaging |
| **Kumar et al.** | 2025 | Gardenfors-style conceptual spaces extractable from LLM prototype embeddings |
| **Chatterjee et al.** | 2023 | BERT and GPT-3 implicitly learn Gardenfors quality dimensions |
| **Banaee & Lowry** | 2026 | Conceptual spaces work for purely abstract concepts (chess strategy) |
| **Wheeler & Natarajan** | 2023 | Geometric semantic representations achieve 99.9% communication compression |

**Significance**: Neural networks — including language models — already learn Gardenfors-compatible geometry. Concepts form convex regions in latent space. Gaussians are the natural mathematical primitive for convex density regions. SGS is not imposing an alien structure on language; it's making explicit what transformers already learn implicitly.

### 4.3 The Parameter Correspondence

| 3D Gaussian Splatting | Semantic Gaussian Splatting |
|---|---|
| **Position** μ ∈ ℝ³ | **Position** μ ∈ ℝᵈ — location in semantic space |
| **Covariance** Σ ∈ ℝ³ˣ³ | **Covariance** Σ ∈ ℝᵈˣᵈ — semantic breadth/uncertainty |
| **Opacity** α | **Salience** α — contribution to final meaning |
| **Color (SH)** | **Features** f — context-dependent semantic attributes |
| **Scene** | **Vocabulary/Knowledge** — collection of semantic Gaussians |
| **Camera viewpoint** | **Query/Context** — semantic viewpoint (see 4.4) |
| **Depth ordering** | **Positional ordering** — sequence-position compositing |

### 4.4 What Is a "Semantic Viewpoint"? — Formal Definition

A **semantic viewpoint** is a projection operator P: ℝᵈ → ℝᵐ (m < d) combined with a query position q ∈ ℝᵐ:

```
Viewpoint V = {P ∈ ℝᵐˣᵈ, q ∈ ℝᵐ}
```

Gaussians are projected: μ' = Pμ, Σ' = PΣP^T. The rendering equation operates on projected Gaussians — exactly as 3DGS projects 3D→2D.

**How this differs from attention**: Attention selects via dot-product similarity in the same space. SGS projects into a *different subspace* — a dimensional reduction, not just a weighting.

### 4.5 Theoretical Convergence: Attention = Energy Minimization = Kernel Evaluation

Three independent lines of research converge to support SGS's rendering-as-computation thesis:

**1. Attention IS energy minimization (Ramsauer et al., ICLR 2021).** "Hopfield Networks is All You Need" proved that the transformer attention mechanism is mathematically equivalent to the update rule of modern continuous Hopfield networks — energy minimization in a field of stored patterns. This means attention is already implicitly doing what SGS proposes explicitly: querying a continuous field to retrieve and compose stored representations.

**2. Attention IS a kernel function (Katharopoulos et al., ICML 2020; Choromanski et al., ICLR 2021).** Linear attention and Performers decompose softmax attention as a kernel function. Hedgehog (Zhang et al., ICLR 2024) showed that the *shape* of the attention kernel can be learned, and that different kernels suit different tasks. **SGS proposes the Gaussian (RBF) kernel as the principled choice** — grounded in spatial geometry and with learnable parameters (mean, covariance).

**3. Transformers and SSMs are dual (Dao & Gu, ICML 2024).** Mamba-2 proved a deep theoretical connection between attention and continuous state-space dynamics — they are two views of the same computation. This means continuous-field representations (like SGS) are not a departure from attention but a different realization of the same underlying operation.

**What SGS adds:** These papers prove attention is already a kernel/energy/field operation, but implicitly so. SGS makes the field explicit — each Gaussian is a named, inspectable, geometrically positioned primitive in the energy landscape. The rendering equation becomes a specific instantiation of the Hopfield energy minimization with Gaussian kernels and transmittance-gated weights.

**Novel contribution:** No paper unifies alpha-compositing, Gaussian kernels, and Hopfield energy minimization into a single framework for language. The formal connection:

- Volume rendering: C = Σ(T_i · α_i · c_i), where T_i = Π(1 − α_j)
- Attention: y = Σ(softmax(q · k_i) · v_i)
- Hopfield: x_new = Σ(softmax(β · x^T · ξ_i) · ξ_i)

All three are weighted sums with different weight computation schemes. SGS proposes a fourth: Gaussian kernel evaluation with transmittance gating — combining the spatial locality of Gaussians with the occlusion modeling of volume rendering.

---

## 5. Proposed Architecture: Semantic Gaussian Splatting (SGS)

### 5.1 Core Representation

```
G = {μ, Σ, α, f, t}
```

- **μ ∈ ℝᵈ** — mean: central meaning
- **Σ ∈ ℝᵈˣᵈ** — covariance: semantic breadth (low-rank factored)
- **α ∈ [0, 1]** — salience
- **f ∈ ℝᵏ** — feature vector
- **t** — positional/temporal parameter

### 5.2 Dual-Space Architecture

**The dimensionality problem**: In d=768, Gaussian evaluation is numerically degenerate (exp(-384) ≈ 0).

**Resolution**: SGS uses two spaces:

1. **Splatting space** (d_s = 32-64): Rendering equation operates here. Numerically tractable.
2. **Feature space** (d_f = 256-768): Rich semantic content per Gaussian.

```
Meaning(q) = Σᵢ fᵢ · αᵢ · K(q, μᵢ, Σᵢ) · Tᵢ
```

Where K is a normalized Gaussian kernel with temperature τ in the low-d splatting space.

**Literature support**: Wheeler & Natarajan (2023) demonstrated that geometric semantic representations achieve massive compression (99.9% rate reduction) — confirming that meaning lives on low-dimensional manifolds even when embedded in high-d spaces.

### 5.3 Vocabulary as a Gaussian Scene

```
V = {G₁, G₂, ..., Gₙ}
```

Polysemous words map to multiple Gaussians:
```
G_"bank" = {G_bank_financial, G_bank_river, G_bank_verb}
```

**Literature support**: Athiwaratkun & Wilson (2017) demonstrated that Gaussian mixtures per word capture polysemy effectively. Chen et al. (GMSG, 2015) showed these mixtures can dynamically adjust component count — a direct precedent for SGS's adaptive density control.

### 5.4 The Semantic Rendering Equation

**Input**: tokens [w₁, ..., wₙ] → activate Gaussians with positional modulation:
```
Gᵢ(pos) = {μᵢ + δ(pos), Σᵢ · σ(pos), αᵢ · relevance(pos), fᵢ + φ(pos)}
```

**Rendering**:
```
Meaning(q) = Σᵢ fᵢ · αᵢ · K(q, μᵢ, Σᵢ) · Tᵢ
Tᵢ = ∏ⱼ₌₁ⁱ⁻¹ (1 − αⱼ · K(q, μⱼ, Σⱼ))    [j ordered by sequence position]
```

### 5.5 Multi-Pass Iterative Rendering

SGS uses P rendering passes, each refining the Gaussian scene:

```
For pass p = 1, ..., P:
  1. Render: Meaning_p(q) from current scene
  2. Update: μᵢ^(p+1) = μᵢ^(p) + Δμ(Meaning_p, fᵢ)
             αᵢ^(p+1) = αᵢ^(p) · gate(Meaning_p, fᵢ)
  3. FFN: fᵢ^(p+1) = FFN(fᵢ^(p), Meaning_p(μᵢ))
```

**Literature support**: Kiruluta (2026) models in-context learning as sequential Bayesian estimation where "adaptation is driven by posterior covariance collapse" — formally related to SGS's multi-pass rendering where Gaussian parameters converge across passes.

### 5.6 Operator Gaussians for Non-Monotonic Composition

Alpha-blending cannot negate or quantify. Operator Gaussians transform the rendering process:

1. **Negation**: Inverts sign of next content Gaussian's feature contribution
2. **Quantifiers**: Modulate scope of subsequent Gaussians (broaden/narrow covariance)
3. **Scope brackets**: Save/restore rendering state for nested clauses

**Honest assessment**: This remains the weakest component. At this point, SGS is a Gaussian rendering framework with operator extensions for ~15% of linguistic phenomena that require non-monotonic composition.

### 5.7 Autoregressive Generation

```
For output position t = 1, ..., T:
  1. Render: meaning_t from scene ∪ {G_output_1, ..., G_output_{t-1}}
  2. Decode: token_t = argmax(meaning_t · W_vocab)
  3. Activate: G_output_t with positional modulation
  4. Add to scene
```

### 5.8 Adaptive Density Control

- **Splitting**: High gradient + large Gaussian → split into specific sub-Gaussians
- **Pruning**: Near-zero opacity → remove
- **Cloning**: Under-represented → duplicate

**Literature support**: Chen et al. (GMSG, 2015) demonstrated dynamic sense component adjustment during training — the NLP-specific precedent for this mechanism. In GMSG, a single word Gaussian adaptively splits into multiple sense Gaussians when training loss indicates unresolved polysemy. SGS generalizes this from word-level sense splitting to vocabulary-level semantic region splitting.

### 5.9 Multi-Scale Semantic Resolution

| Scale | Linguistic Unit | Gaussian Properties |
|---|---|---|
| **Macro** | Concepts, themes, topics | Large Σ, high α |
| **Meso** | Words, phrases | Medium Σ, variable α |
| **Micro** | Morphemes, characters | Small Σ, low α |

**Literature support**: Athiwaratkun, Wilson & Anandkumar (2018) combined subword (FastText) composition with Gaussian mixtures, demonstrating that multi-scale Gaussian representations work in practice. Gaussian LDA (Das et al., 2015) operates at the macro (topic) scale. Word2Gauss at the meso scale. SGS unifies all scales in a single rendering framework.

---

## 6. Theoretical Analysis

### 6.1 Core Assumptions — Updated with Literature Evidence

| # | Assumption | Status | Evidence |
|---|---|---|---|
| A1 | Language meaning embeds in continuous geometric space | **Strongly validated** | Word2Vec (2013), GloVe (2014), BERT (2019), Tetkova (2023): convex regions in neural net latent spaces across modalities |
| A2 | Gaussians are more natural than points for meaning | **Validated (word level)** | 11-year trajectory: Vilnis (2015) → Athiwaratkun (2017, 2018) → Yuksel (2021) → Huang (2026). Consistent advantages for uncertainty, entailment, polysemy |
| A3 | Composition via Gaussian rendering beats simple blending | **Untested — CORE RISK** | GaussCSE (2023) validates sentence-level Gaussians exist. Gaussian LDA validates topic-level composition. The *rendering equation* as the composition operator is novel |
| A4 | Low-d splatting preserves semantic structure | **Supported** | Wheeler (2023): 99.9% compression via geometric semantics. Conceptual spaces operate in low dimensions by design |
| A5 | Multi-pass rendering = sufficient depth | **Plausible** | Kiruluta (2026): ICL as sequential Bayesian estimation with covariance collapse |
| A6 | Operator Gaussians handle non-monotonic composition | **Speculative** | No direct literature support. Weakest assumption |

### 6.2 Potential Advantages — Literature-Grounded

| Advantage | Mechanism | Confidence | Supporting Evidence |
|---|---|---|---|
| **Explicit uncertainty** | Covariance encodes semantic breadth | **High** | Vilnis (2015), Athiwaratkun (2017, 2018), Yuksel (2021), Cui/Prompt2Gaussia (2023) |
| **Interpretability** | Per-Gaussian contributions traceable in low-d space | **Medium** | Conceptual spaces literature shows geometric representations are human-interpretable (Gardenfors, Wheeler) |
| **Compositional structure** | Rendering equation as defined composition operator | **Medium** | Gaussian LDA composes topic Gaussians; WMD uses OT for distributional comparison; rendering equation is novel for language |
| **Adaptive resolution** | Split/prune/clone allocates capacity | **Medium-High** | GMSG (Chen et al., 2015) demonstrates dynamic sense splitting in NLP; 3DGS validates in vision |
| **Sub-quadratic computation** | Locality in low-d splatting space | **Medium** | Depends on effective sparsity at d=64 |
| **Hierarchical representation** | Hyperbolic Gaussians for taxonomy | **Medium** | Nagano et al. (2019): pseudo-hyperbolic Gaussians with differentiable parameters |

### 6.3 Risks — Updated

| # | Risk | Severity | Mitigation | Literature Insight |
|---|---|---|---|---|
| R1 | Training signal mismatch | **Critical** | Gradient accumulation; auxiliary losses | SIREN (Sitzmann, 2020) shows neural field training generalizes beyond pixel loss to wavefields, sound, PDEs |
| R2 | Alpha-blending can't negate/quantify | **Critical** | Operator Gaussians | No literature support — remains weakest point |
| R3 | High-d numerical degeneration | **Resolved** | Low-d splatting space | Wheeler (2023): geometric semantics compress 99.9% |
| R4 | No depth in single pass | **Resolved** | Multi-pass rendering | Kiruluta (2026): ICL as sequential Bayesian estimation |
| R5 | Word2Gauss didn't scale | **Mitigated** | Composition-level test | Post-Word2Gauss lineage (2017-2026) shows continued productivity; issue was benchmarking, not fundamental limits |
| R6 | Anisotropic raw embeddings | **New risk** | Learned splatting space | Li/BERT-flow (2020): raw transformer embeddings are non-Gaussian. SGS must learn its own geometry |

---

## 7. Experimental Validation Plan

### Phase 0: Numerical Feasibility (Weeks 1-2)

Initialize 1,000 Gaussians from GloVe→PCA(d=64). Verify non-trivial kernel evaluations and gradient flow.

**Falsification**: If >90% evaluations underflow, try inverse-quadratic kernel or reduce to d=16.

### Phase 1: Gaussian Composition (Months 1-3)

GloVe 300d → PCA to d=64 splatting + 300d features. Train on STS-B with multi-pass rendering (P=4).

**Success**: STS-B Spearman ≥ 0.78 (beating SIF). Ablation: rendering > mean-pooling; multi-pass > single-pass.

**Falsification**: If below SIF, pivot to hybrid A1/A3.

### Phase 2: Viewpoints + Operators (Months 3-6)

**2a**: Multi-task (sentiment, topic, NER) via different viewpoint projections.

**2b**: Monotonicity NLI, SCAN, COGS for negation/quantification/compositionality.

### Phase 3: Adaptive Density (Months 6-9)

10K initial Gaussians → language modeling with split/prune/clone every 1K steps. Compare adaptive vs. fixed.

### Phase 4: Full SGS Language Model (Months 9-18)

10M parameters, 50K Gaussians, d_s=64, d_f=512, P=8 passes. Autoregressive. Benchmark against equivalent-size transformer.

### Parallel Tracks

**A1 — Gaussian Transformer**: Drop-in replacement of softmax attention with Bhattacharyya-coefficient Gaussian attention. Tests "do Gaussian primitives help?" independently.

**A3 — Gaussian Mixture Attention**: Replace softmax with Gaussian mixture evaluation. Minimal modification.

---

## 8. What Would Success Look Like?

### 8.1 Minimum Viable Result

1. Phase 0 passes (numerical feasibility at d=64)
2. Phase 1: STS-B ≥ 0.78 with rendering > mean-pooling
3. Phase 2a: Competitive on one task via viewpoint selection
4. Phase 2b: Above-chance on negation/quantification

### 8.2 Maximum Upside

A language model that is more interpretable, better at compositional generalization, natively uncertainty-aware, and competitive at small scale.

### 8.3 What Failure Teaches

| Failure Mode | Next Step |
|---|---|
| Phase 0: numerical collapse | Cauchy/Student-t kernels; d=16 |
| Phase 1: rendering = averaging | Pivot to hybrid A1/A3 |
| Phase 2b: operators fail | SGS for monotonic only; transformers for logic |
| Phase 4: quality OK but slow | SGS = interpretability tool, not speed tool |

---

## 9. Novel Contributions — Confirmed by Literature Gap Analysis

The literature review confirms these as **genuinely novel**:

| # | Contribution | Why Novel |
|---|---|---|
| 1 | **Gaussian splatting for pure language modeling** | All existing work either uses Gaussians for pairwise word similarity OR embeds language into 3D Gaussian scenes. No one applies the splatting rendering framework to language itself. |
| 2 | **Rendering equation as semantic composition** | GaussCSE shows sentences can be Gaussians. Gaussian LDA shows topics can be Gaussians. How word Gaussians *compose into* sentence Gaussians via rendering is new. |
| 3 | **Bridging Gardenfors' conceptual spaces with Gaussian splatting** | Conceptual spaces predict convex regions; Gaussians are the natural primitive; splatting is the rendering engine. This triad has never been connected. |
| 4 | **Formal alpha-compositing ↔ attention theorem** | The structural parallel is implicit in many architectures but formally unproven. |
| 5 | **Adaptive density control for semantic vocabulary** | GMSG does dynamic sense splitting, but not within a rendering/splatting framework with split/prune/clone. |

---

## 10. Related Work and Literature

### 10.1 3D Gaussian Splatting

- **Kerbl et al. (2023)** — 3DGS. SIGGRAPH/TOG.
- **Mildenhall et al. (2020)** — NeRF. ECCV 2020.
- **Zwicker et al. (2002)** — EWA Splatting. IEEE TVCG.
- **Wu et al. (2024)** — 4D Gaussian Splatting. CVPR 2024.

### 10.2 Gaussian Splatting Beyond Vision

- **Kim et al. (2026)** — Station2Radar: Gaussian splatting for precipitation fields.
- **Cai et al. (2026)** — GaussTwin: Gaussian digital twins for robotics.
- **Xiong et al. (2025)** — Mind-to-Face: EEG→Gaussian avatars.
- **Luo et al. (2026)** — DynFOA: Gaussian splatting for spatial audio.
- **Li et al. (2025)** — PIN-WM: Physics-informed Gaussian world models.

### 10.3 Language-Embedded Gaussians

- **Qin et al. (2024)** — LangSplat. CVPR 2024.
- **Li et al. (2025)** — LangSplatV2: 450+ FPS.
- **Li et al. (2025)** — 4D LangSplat: dynamic semantic Gaussians.
- **Maggio & Carlone (2025)** — Bayesian Fields: task-driven semantic Gaussians.
- **Kerr et al. (2023)** — LERF. ICCV 2023 (Oral).
- **Zhou et al. (2024)** — Feature 3DGS.
- **Barhdadi et al. (2026)** — 4D Motion-Language Gaussian fields.

### 10.4 Gaussian Word Embeddings (2015-2026)

- **Vilnis & McCallum (2015)** — Word2Gauss. ICLR 2015.
- **Athiwaratkun & Wilson (2017)** — Multimodal Word Distributions. ACL 2017.
- **Athiwaratkun & Wilson (2018)** — Hierarchical Density Order Embeddings. ICLR 2018.
- **Athiwaratkun, Wilson & Anandkumar (2018)** — Probabilistic FastText. ACL 2018.
- **Chen et al. (2015)** — GMSG: Dynamic Gaussian Mixture Skip-gram.
- **Yuksel et al. (2021)** — Semantic Change Detection with Gaussian Embeddings. IEEE/ACM TASLP.
- **Huang (2026)** — Gaussian Joint Embeddings. arXiv.

### 10.5 Sentence/Document-Level Gaussians

- **Yoda et al. (2023)** — GaussCSE: Sentence Gaussians. EACL 2024.
- **Das et al. (2015)** — Gaussian LDA. ACL 2015.
- **Li et al. (2020)** — BERT-flow. EMNLP 2020.
- **Yurochkin et al. (2019)** — Hierarchical OT for documents. NeurIPS 2019.

### 10.6 Optimal Transport in NLP

- **Kusner et al. (2015)** — Word Mover's Distance. ICML 2015.
- **Wu et al. (2018)** — Word Mover's Embedding. NeurIPS 2018.

### 10.7 Uncertainty via Distributional Representations

- **Cui et al. (2023)** — Prompt2Gaussia. arXiv.
- **Kiruluta (2026)** — Bayesian Kalman View of ICL. arXiv.

### 10.8 Hyperbolic Gaussians

- **Nagano et al. (2019)** — Wrapped Normal on Hyperbolic Space. ICML 2019.
- **Nickel & Kiela (2017)** — Poincare Embeddings. NeurIPS 2017.

### 10.9 Conceptual Spaces — Computational Models

- **Gärdenfors (2000, 2014)** — Conceptual Spaces theory.
- **Rosch (1973-1978)** — Prototype theory.
- **Kumar et al. (2025)** — Extracting Conceptual Spaces from LLMs.
- **Fel et al. (2025)** — Minkowski Representation Hypothesis.
- **Tetkova et al. (2023)** — Convex Decision Regions in Deep Networks.
- **Wheeler & Natarajan (2023)** — Semantic Communication via Geometry of Meaning.
- **Tull et al. (2024)** — From Conceptual Spaces to Quantum Concepts.
- **Banaee & Lowry (2026)** — Abstract Concept Modelling in Conceptual Spaces.

### 10.10 Differentiable Rendering as General Computation

- **Sitzmann et al. (2020)** — SIREN. NeurIPS 2020.
- **Kato et al. (2020)** — Differentiable Rendering Survey.
- **Mostafa et al. (2026)** — Differentiable Rendering for Tabular Data.

### 10.11 Alpha-Compositing / MoE / Attention Connections

- **Jin et al. (2025)** — MoE-GS: Mixture of Experts for Gaussian Splatting.
- **D'Amicantonio et al. (2025)** — GS-MoE: Gaussians Guide Expert Routing.
- **Liang et al. (2023)** — ReTR: Rendering via Transformer.

### 10.12 State Space Models and Linear Attention

- **Gu & Dao (2023)** — Mamba: Selective State Spaces. Linear-time, matches 2x-sized transformers.
- **Gu, Goel & Re (2021)** — S4: Structured State Spaces. ICLR 2022.
- **Dao & Gu (2024)** — Mamba-2: Transformers are SSMs (duality proof). ICML 2024.
- **Poli et al. (2023)** — Hyena Hierarchy. ICML 2023.
- **Peng et al. (2023)** — RWKV: Linear attention with exponential decay. EMNLP 2023.
- **Katharopoulos et al. (2020)** — Linear Attention: Transformers are RNNs. ICML 2020.
- **Zhang et al. (2024)** — Hedgehog: Learnable linear attention kernels. ICLR 2024.

### 10.13 Diffusion Models for Text

- **Li et al. (2022)** — Diffusion-LM. NeurIPS 2022.
- **Dieleman et al. (2022)** — CDCD: Continuous Diffusion for Categorical Data.
- **Sahoo et al. (2024)** — MDLM: Masked Diffusion Language Models. NeurIPS 2024.
- **Lou et al. (2023)** — SEDD: Score Entropy Discrete Diffusion. ICML 2024.

### 10.14 Energy-Based Models and Geometric Deep Learning

- **Ramsauer et al. (2020)** — Hopfield Networks is All You Need (attention = energy minimization). ICLR 2021.
- **Deng et al. (2020)** — Residual EBMs for Text Generation. ICLR 2020.
- **Bronstein et al. (2021)** — Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.

### 10.15 Transformers and Compositional Generalization

- **Vaswani et al. (2017)** — Attention Is All You Need.
- **Devlin et al. (2019)** — BERT.
- **Ethayarajh (2019)** — Contextual embedding geometry.
- **Lake & Baroni (2018)** — SCAN benchmark.
- **Kaplan et al. (2020)** — Scaling laws.
- **Power et al. (2022)** — Grokking.
- **Choromanski et al. (2021)** — Performers (FAVOR+). ICLR 2021.
- **Balestriero et al. (2021)** — High-d learning is always extrapolation.
- **Drozdov et al. (2022)** — Compositional Parsing with LLMs. ICLR 2023.
- **Anil et al. (2022)** — Length Generalization in LLMs.

---

## 11. Recommendations and Next Steps

### 11.1 Immediate Actions

1. **Phase 0** (1-2 weeks): Numerical feasibility in d=64
2. **Composition prototype** (2-4 weeks): Multi-pass rendering on STS-B
3. **Parallel hybrids**: Gaussian Transformer (A1) and Gaussian Mixture Attention (A3)
4. **Formal theorem**: Prove the alpha-compositing ↔ attention connection mathematically

### 11.2 Research Partnerships

- **3DGS researchers**: Differentiable rasterization
- **Geometric deep learning**: Non-Euclidean representations
- **Computational linguistics**: Evaluation methodology
- **Conceptual spaces researchers**: Gardenfors-compatible evaluation (Kumar, Schockaert groups)

### 11.3 Intellectual Property

- Research publication establishing priority
- Patent on SGS architecture + rendering equation for language
- Open-source prototype

---

## 12. Conclusion

Semantic Gaussian Splatting proposes that the revolution 3DGS brought to vision can be adapted for language. Following adversarial challenge and comprehensive literature review, we are now on firmer ground:

**What the literature validates:**
- Gaussians outperform points for word meaning (11-year trajectory, 7+ papers)
- Sentences can be represented as Gaussians (GaussCSE, 2023)
- Gaussian splatting generalizes beyond vision (weather, robotics, audio, brain signals)
- Neural networks already learn Gardenfors-compatible convex regions (3+ empirical papers)
- Adaptive sense splitting works in NLP (GMSG, 2015)
- Low-dimensional geometric representations preserve semantic structure (Wheeler, 2023)

**What remains untested:**
- The rendering equation as a composition mechanism for language (CORE NOVELTY)
- Operator Gaussians for negation/quantification (WEAKEST COMPONENT)
- Scaling behavior of the full architecture

**What is confirmed as novel:**
- Applying Gaussian splatting to pure language modeling
- The rendering equation as semantic composition
- Bridging Gardenfors' conceptual spaces with splatting
- Formal alpha-compositing ↔ attention theorem

The architecture is honestly a hybrid: Gaussian rendering within layers, iterative refinement across layers, operator extensions for logical language. This is less theoretically pure than "meaning is rendering" but grounded in a decade of evidence that Gaussian representations are natural for language.

The fundamental bet: **explicit, geometric, interpretable primitives are a better substrate for language than implicit, learned, opaque attention patterns.** The literature says this bet is worth making. The experiments will determine if it pays off.

---

## Appendix A: Orthogonal Challenge FAQ

*(Unchanged from v3 — see v2_challenge.md for full adversarial review)*

| # | Challenge | Resolution | Status |
|---|---|---|---|
| C1 | No dense training signal | Gradient accumulation; SIREN shows field training generalizes | Partially resolved |
| C2 | Can't negate/quantify | Operator Gaussians | Weakest component |
| C3 | High-d degenerate | Low-d splatting space | Resolved |
| C4 | No depth | Multi-pass rendering | Resolved (hybrid) |
| C5 | Word2Gauss didn't scale | Post-W2G lineage shows continued value; SGS tests composition | Mitigated |
| C6 | Viewpoint undefined | Formal definition (projection + query) | Resolved |
| C7 | No depth ordering | Sequence position | Resolved |
| C8 | NAR fails | Autoregressive adopted | Resolved |
| C9 | Prototype theory limited | Conceptual spaces validated computationally for abstract concepts | Strengthened |
| C10 | Interpretability unproven | Planned for Phase 2 | Open |
| C11 | Efficiency incomplete | Phase 0 | Deferred |
| C12 | Bar too low | Raised to ≥ 0.78 | Resolved |

---

*This whitepaper is an experimental proposal strengthened by a comprehensive literature review of 70+ papers. It positions SGS within a decade of Gaussian language research and a broader landscape of alternative architectures (SSMs, diffusion, energy-based models, kernel attention), identifies five confirmed novel contributions, and specifies falsifiable experiments. The goal remains: determine whether Gaussian primitives and rendering-based composition offer genuine advantages for language understanding.*
