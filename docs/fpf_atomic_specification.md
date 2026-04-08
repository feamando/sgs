# SGS Atomic Specification — FPF Structured Reasoning

**FPF Cycle:** SGS-ATOMS-2026-04-07
**Phase:** Complete (0→1→2→3→4→5)
**Goal:** Strictly define every atomic component of SGS with formal math, constraints, and falsifiable predictions

---

## Phase 0: Bounded Context

### Domain Vocabulary

| Term | Formal Definition | Constraints |
|---|---|---|
| **Semantic Space** | A learned metric space S = (ℝ^d_s, g) where g is a Riemannian metric (Euclidean in v1, potentially hyperbolic in v2) | d_s ∈ {32, 64}; must support non-degenerate Gaussian evaluation |
| **Feature Space** | A vector space F = ℝ^d_f carrying rich semantic content | d_f ∈ {256, 512, 768} |
| **Semantic Gaussian** | A tuple G = (μ, L, α, f, t) — the atomic primitive | See Atom A1 |
| **Gaussian Scene** | An ordered multiset Ω = {G₁, ..., Gₙ} of activated Semantic Gaussians | n = number of active tokens + generated tokens |
| **Rendering Pass** | A single evaluation of the rendering equation over Ω from a query point | Produces one d_f-dimensional meaning vector |
| **Query Viewpoint** | A tuple V = (P, q) specifying projection and position | See Atom A4 |

### Invariants

1. **Positive semi-definiteness**: Σ must be PSD at all times → enforced by factored parameterization
2. **Opacity bound**: α ∈ [0, 1] → enforced by sigmoid activation
3. **Transmittance monotonicity**: T_i ≥ T_{i+1} for sequential compositing (transmittance can only decrease)
4. **Gradient flow**: Every parameter must receive non-zero gradients from the loss function
5. **Numerical stability**: No operation produces NaN or Inf in float32 at d_s ≤ 64

---

## Phase 1: Hypotheses — Atomic Components

Seven atomic components identified. Each is stated as a hypothesis (H) with a formal specification.

---

### ATOM A1: The Semantic Gaussian Primitive

**Hypothesis H1:** A single word/concept can be faithfully represented as a parameterized Gaussian distribution in semantic space, where the parameters encode distinct aspects of meaning.

**Formal Specification:**

```
G = (μ, L, α, f, t)
```

| Parameter | Type | Dimensions | Definition | Semantic Role |
|---|---|---|---|---|
| **μ** | Vector | ℝ^d_s | Mean position in splatting space | Core denotational meaning |
| **L** | Lower-triangular | ℝ^d_s × d_s | Cholesky factor: Σ = LL^T | Semantic breadth, directional uncertainty. Diagonal of L = scale per dimension; off-diagonal = correlations between semantic dimensions |
| **α** | Scalar | [0, 1] | Base salience (sigmoid-activated) | Default importance/contribution weight |
| **f** | Vector | ℝ^d_f | Feature vector in feature space | Rich semantic attributes: POS, sentiment, register, domain, entailment features |
| **t** | Scalar | ℝ | Positional/temporal index | Sequence position (discrete) or continuous time |

**Parameter Count Per Gaussian:**
- μ: d_s = 64
- L: d_s(d_s+1)/2 = 2,080 (lower triangular)
- α: 1
- f: d_f = 512
- t: 1
- **Total: 2,658 parameters per Gaussian**

**Covariance Recovery:**
```
Σ = LL^T + εI    (ε = 1e-6 for numerical stability)
```

**Why Cholesky, not rotation+scale:** In d_s=64, a rotation matrix has 2,016 DOF (Givens rotations) plus 64 scales = 2,080 total — same count as Cholesky. But Cholesky is simpler to implement, naturally PSD, and standard in probabilistic ML. Rotation+scale is preferred in 3DGS only because d=3 makes quaternions natural.

**Falsifiable Prediction:** Trained Gaussian covariances will show interpretable structure:
- Rare words → larger det(Σ) (broader uncertainty)
- Concrete nouns → smaller det(Σ) than abstract nouns
- Hypernyms → larger det(Σ) than hyponyms (Athiwaratkun & Wilson, 2018)

**Evidence:** Vilnis & McCallum (2015), Athiwaratkun & Wilson (2017, 2018), Yuksel et al. (2021) — all validate that Gaussian parameters encode meaningful linguistic properties.

**Assurance Level:** L2 (validated by multiple independent empirical studies)

---

### ATOM A2: The Normalized Gaussian Kernel

**Hypothesis H2:** A temperature-scaled Gaussian kernel in d_s dimensions produces numerically stable, meaningful similarity values between query points and Gaussian centers.

**Formal Specification:**

```
K(q, μ, Σ) = exp(-½ · D_M(q, μ, Σ) / τ)
```

Where:
- **D_M(q, μ, Σ) = (q - μ)^T Σ^{-1} (q - μ)** is the Mahalanobis distance
- **τ > 0** is a learned temperature parameter

**Implementation (numerically stable):**
```
v = L^{-1}(q - μ)          # Solve triangular system: O(d_s²)
D_M = v^T v                 # Squared Mahalanobis distance
K = exp(-0.5 · D_M / τ)    # Temperature-scaled kernel
```

**Temperature Semantics:**
- τ = 1: standard Gaussian kernel (too peaked in high d)
- τ = d_s: normalizes expected distance (E[D_M] = d_s for points drawn from the Gaussian)
- τ learned: model decides effective radius

**Numerical Analysis at d_s = 64:**
- E[D_M] for a sample FROM the Gaussian = 64
- With τ = 64: E[K] = exp(-0.5) ≈ 0.607 — well-behaved
- For a point 2σ away: D_M ≈ 256, K = exp(-2) ≈ 0.135 — meaningful gradient
- For a point 4σ away: D_M ≈ 1024, K = exp(-8) ≈ 3.4e-4 — effectively sparse

**Sparsity Threshold:** Define effective support as K > ε_sparse (e.g., ε_sparse = 1e-3). At d_s=64, τ=64, this gives a radius of ~4σ. Only Gaussians within this radius contribute meaningfully.

**Falsifiable Prediction:** At d_s=64 with learned τ, >80% of kernel evaluations will be below ε_sparse = 1e-3 after training (demonstrating effective sparsity). If <50% are sparse, the locality advantage over attention is marginal.

**Evidence:** Phase 0 experiment will directly test this.

**Assurance Level:** L1 (logically sound; empirical validation pending Phase 0)

---

### ATOM A3: The Transmittance-Gated Compositing Equation

**Hypothesis H3:** Front-to-back alpha-compositing with transmittance gating, ordered by sequence position, produces compositionally meaningful sentence representations from word Gaussians.

**Formal Specification:**

Given an ordered Gaussian scene Ω = {G_1, ..., G_n} (ordered by sequence position) and a query point q ∈ ℝ^d_s:

```
Meaning(q) = Σᵢ₌₁ⁿ wᵢ · fᵢ

where:
  wᵢ = αᵢ · K(q, μᵢ, Σᵢ) · Tᵢ                    [blending weight]
  Tᵢ = ∏ⱼ₌₁ⁱ⁻¹ (1 − αⱼ · K(q, μⱼ, Σⱼ))          [accumulated transmittance]
  T₁ = 1                                             [first token has full transmittance]
```

**Properties (mathematically provable):**
1. **Bounded output:** Σᵢ wᵢ ≤ 1 (each weight is non-negative; total bounded by 1 - T_final)
2. **Monotonic transmittance:** T₁ ≥ T₂ ≥ ... ≥ Tₙ ≥ 0
3. **Differentiable:** All operations are smooth and differentiable w.r.t. all parameters
4. **Order-dependent:** Changing token order changes the output (captures word order)
5. **Early tokens dominate:** T_i decreases with i, giving earlier tokens natural advantage (mitigated by multi-pass, Atom A5)

**Gradient Flow:**
```
∂Meaning/∂fᵢ = wᵢ · I                               [direct, always non-zero if wᵢ > 0]
∂Meaning/∂αᵢ = K(q,μᵢ,Σᵢ) · Tᵢ · fᵢ              [via blending weight]
                + Σⱼ>ᵢ [∂wⱼ/∂αᵢ] · fⱼ              [via transmittance of later tokens]
∂Meaning/∂μᵢ = via ∂K/∂μᵢ, propagating through wᵢ and Tⱼ for j>i
```

Every parameter receives gradient from both its own blending weight AND its effect on all later tokens' transmittance.

**Critical Limitation — Monotonic Composition Only:**
This equation can only ADD features (weighted sum with non-negative weights). It CANNOT:
- Negate (require negative weights → Atom A6)
- Quantify over scope (require scope tracking → Atom A6)
- Recurse (require stack → Atom A6)

~85% of natural language is monotonic composition (adjective+noun, verb+object, modifier+head). The remaining ~15% requires Atom A6.

**Falsifiable Prediction:** Rendering-based composition will outperform mean-pooling of Gaussian means on STS-B by ≥ 0.05 Spearman. If not, the compositing mechanism adds nothing beyond the Gaussian representation itself.

**Evidence:** No direct precedent. Gaussian LDA (Das et al., 2015) composes topic Gaussians; WMD (Kusner et al., 2015) compares distributional representations. Neither uses alpha-compositing.

**Assurance Level:** L0 (untested — CORE NOVELTY)

---

### ATOM A4: The Semantic Viewpoint (Projection + Query)

**Hypothesis H4:** Projecting Gaussians into task-specific subspaces before rendering enables extraction of different semantic aspects from the same scene.

**Formal Specification:**

```
Viewpoint V = (P, q)
  P ∈ ℝ^(d_m × d_s)     — projection matrix (learned, per-task or per-head)
  q ∈ ℝ^d_m              — query position in projected space
  d_m ≤ d_s              — projected dimensionality
```

**Gaussian Projection (EWA Splatting in semantic space):**
```
μ'ᵢ = P · μᵢ                              [projected mean]
Σ'ᵢ = P · Σᵢ · P^T                        [projected covariance]
K'(q, μ'ᵢ, Σ'ᵢ) = exp(-½ · D_M' / τ)    [kernel in projected space]
```

This is exactly the EWA splatting projection (Zwicker et al., 2002) from d_s to d_m instead of 3D to 2D.

**Multi-View Rendering (analogous to multi-head attention):**
```
For head h = 1, ..., H:
  Meaning_h(q_h) = RenderingEquation(Ω, P_h, q_h)

MultiView(Ω) = Concat(Meaning_1, ..., Meaning_H) · W_out
```

Each head has its own projection P_h and query q_h. This provides H different "views" of the same Gaussian scene.

**Relationship to attention:**
| Property | Multi-Head Attention | Multi-View SGS Rendering |
|---|---|---|
| Weight computation | softmax(QK^T/√d) | Gaussian kernel + transmittance |
| Value composition | Linear combination of V | Alpha-composited features |
| Per-head projection | W_Q, W_K, W_V ∈ ℝ^(d×d_h) | P ∈ ℝ^(d_m×d_s), q ∈ ℝ^d_m |
| Locality | Global (all tokens attend to all) | Local (only nearby Gaussians contribute) |

**Falsifiable Prediction:** Different learned projections P_h will specialize for different semantic aspects (verifiable by probing: sentiment heads, syntactic heads, topic heads). If all heads converge to similar projections, the multi-view mechanism is redundant.

**Evidence:** Multi-head attention already shows head specialization (Clark et al., 2019). The question is whether projection-based rendering shows similar or different specialization.

**Assurance Level:** L1 (logically sound; multi-head specialization validated in attention, untested in rendering)

---

### ATOM A5: Iterative Semantic Rendering (Multi-Pass)

**Hypothesis H5:** Multiple rendering passes with inter-pass Gaussian parameter updates provide sufficient depth-of-computation for complex language understanding.

**Formal Specification:**

```
For pass p = 1, 2, ..., P:

  // Step 1: Render (same equation as Atom A3, using current parameters)
  Meaning_p = RenderingEquation(Ω^(p), V)

  // Step 2: Context broadcast — evaluate rendered meaning at each Gaussian's position
  c_i^(p) = RenderingEquation(Ω^(p), (I, μ_i^(p)))    [render at own position]

  // Step 3: Gaussian parameter update
  Δμ_i = MLP_μ(f_i^(p), c_i^(p))                       [position shift]
  μ_i^(p+1) = μ_i^(p) + Δμ_i

  Δα_i = σ(MLP_α(f_i^(p), c_i^(p)))                    [salience gate]
  α_i^(p+1) = α_i^(p) · Δα_i

  // Step 4: Feature update (feedforward)
  f_i^(p+1) = f_i^(p) + FFN(f_i^(p), c_i^(p))          [residual connection]

  // Covariance is NOT updated per-pass (too expensive; learned per-word only)
```

**Parameter count of update MLPs (per pass):**
- MLP_μ: (d_f + d_f) → d_s → d_s = 2·d_f·d_s + d_s² ≈ 69,632
- MLP_α: (d_f + d_f) → 1 = 2·d_f + 1 ≈ 1,025
- FFN: 2·d_f · 4·d_f · 2 (standard transformer FFN shape) ≈ 2,097,152

**Total per pass:** ~2.17M parameters
**Total for P=8 passes:** ~17.3M parameters (comparable to a small transformer)

**What each pass accomplishes:**
- Pass 1: Lexical activation — word Gaussians activate at default positions
- Pass 2: Local disambiguation — "bank" shifts toward financial or river sense based on neighboring Gaussians
- Pass 3: Syntactic integration — subject/object roles encoded via positional shifts
- Pass 4-8: Deeper semantic composition, co-reference, pragmatic context

**Relationship to Diffusion-LM:** Each pass refines the Gaussian scene, analogous to a denoising step in diffusion models (Li et al., NeurIPS 2022). The key difference: SGS refines explicit Gaussian parameters; diffusion refines continuous latent vectors.

**Relationship to Bayesian estimation:** Kiruluta (2026) models ICL as Bayesian filtering where covariance collapses across steps. In SGS, opacity α gates down across passes (Δα < 1 reduces α), and position μ converges — formally similar to posterior concentration.

**Falsifiable Prediction:** Performance must improve monotonically with P up to some saturation point. If P=1 matches P=8, the multi-pass mechanism adds nothing. Expected: significant improvement from P=1 to P=4; diminishing returns from P=4 to P=8.

**Evidence:** Iterative refinement improves NAR translation (Lee et al., 2018). Transformer layer ablations show monotonic improvement up to ~12 layers for base models.

**Assurance Level:** L1 (logically sound; iterative refinement validated elsewhere; SGS-specific validation needed)

---

### ATOM A6: Operator Gaussians (Non-Monotonic Composition)

**Hypothesis H6:** A subset of Gaussians can be designated as operators that modify the rendering process rather than contributing content, enabling negation, quantification, and scope.

**Formal Specification:**

Each Gaussian has a type field:
```
G = (μ, L, α, f, t, type)
  type ∈ {CONTENT, NEGATE, QUANTIFIER, SCOPE_OPEN, SCOPE_CLOSE}
```

**Type-specific rendering behavior:**

```
// During compositing at position i:
switch(G_i.type):

  case CONTENT:
    // Standard rendering (Atom A3)
    contribution_i = w_i · f_i

  case NEGATE:
    // Flip sign of NEXT content Gaussian's contribution
    negate_flag = true

  case QUANTIFIER:
    // Modulate scope of subsequent Gaussians
    scope_transform = MLP_quant(f_i)      // Learned transform
    // Applied to Σ of subsequent Gaussians until scope closes:
    // Σ_j^(scoped) = scope_transform(Σ_j) for j in scope

  case SCOPE_OPEN:
    // Push current transmittance state to stack
    T_stack.push(T_current)

  case SCOPE_CLOSE:
    // Pop transmittance state
    T_current = T_stack.pop()
```

**When negate_flag is true for the next CONTENT Gaussian:**
```
contribution_j = w_j · (-f_j)            // Negated feature contribution
negate_flag = false                        // Reset after one application
```

**Type assignment:** Types are LEARNED, not hardcoded. During training, the model discovers which tokens function as operators. Initialization: function words ("not", "every", "no", "if", "(", ")") initialized as NEGATE/QUANTIFIER/SCOPE types based on POS tags; all others as CONTENT. The type field is a softmax over 5 categories — allowing soft transitions during training.

**Soft type (differentiable):**
```
type_probs = softmax(W_type · f_i)       // [p_content, p_negate, p_quant, p_scope_open, p_scope_close]
contribution_i = p_content · (w_i · f_i)
                + p_negate · (w_i · (-f_{next}))
                + p_quant · scope_modulation(...)
                + ...
```

**Honest assessment:** This is the most complex and least validated atom. The soft-type mechanism makes it differentiable but adds significant complexity. The stack mechanism (scope) requires sequential processing that limits parallelism.

**Falsifiable Predictions:**
1. Models with operator Gaussians outperform models without on Monotonicity NLI by ≥ 5% accuracy
2. Learned type distributions will correlate with linguistic function: "not" → high p_negate; "every" → high p_quant
3. If operator Gaussians show no improvement on negation benchmarks, they should be dropped in favor of handling negation in the FFN (Atom A5, Step 4)

**Evidence:** No direct precedent for operator Gaussians. Negation handling in neural models is an active research area (Hossain et al., 2022; Truong et al., 2023).

**Assurance Level:** L0 (speculative — WEAKEST COMPONENT)

---

### ATOM A7: Adaptive Density Control

**Hypothesis H7:** Dynamic splitting, pruning, and cloning of Gaussians during training produces a self-organizing vocabulary that allocates representational capacity where it's most needed.

**Formal Specification:**

Every N training steps (N = 1000), evaluate each Gaussian:

```
For each G_i in vocabulary V:

  // Metrics (accumulated over N steps):
  grad_norm_i = mean(||∂L/∂μ_i||)         // Positional gradient magnitude
  opacity_i = sigmoid(raw_α_i)             // Current opacity
  scale_i = det(Σ_i)^(1/d_s)             // Geometric mean scale

  // SPLIT: High gradient + large scale → under-specified region
  if grad_norm_i > τ_grad AND scale_i > τ_scale:
    G_a, G_b = split(G_i)
    // G_a.μ = μ_i + ε·v_max    (shift along max eigenvalue direction)
    // G_b.μ = μ_i - ε·v_max
    // G_a.Σ = G_b.Σ = Σ_i / 4  (halve scale in each dimension)
    // G_a.f = G_b.f = f_i       (clone features)
    // G_a.α = G_b.α = α_i       (clone opacity)
    V = V ∪ {G_a, G_b} \ {G_i}

  // CLONE: High gradient + small scale → needs more coverage
  if grad_norm_i > τ_grad AND scale_i ≤ τ_scale:
    G_c = clone(G_i)
    // G_c.μ = μ_i + ε·∇_μ L   (shift toward gradient)
    // G_c.Σ = Σ_i              (same shape)
    V = V ∪ {G_c}

  // PRUNE: Low opacity → irrelevant
  if opacity_i < τ_prune:
    V = V \ {G_i}
```

**Thresholds (hyperparameters):**
| Threshold | Default | Meaning |
|---|---|---|
| τ_grad | 0.01 | Gradient norm threshold for densification |
| τ_scale | median(scales) | Scale threshold for split vs. clone |
| τ_prune | 0.005 | Opacity below which Gaussians are pruned |
| N | 1000 | Steps between density control evaluations |

**Opacity Reset (every 5N steps):**
```
For all G_i: raw_α_i = logit(0.01)    // Reset to near-zero
// Forces model to re-justify every Gaussian's existence
// Gaussians that matter will quickly regain opacity; noise will stay low
```

**Vocabulary Growth Model:**
- Initial: |V| = |vocab| (one Gaussian per token, ~50K)
- After training: expect |V| ≈ 1.5-3x initial (splits for polysemy, pruning for rare tokens)
- The final |V| is a LEARNED property of the training data, not a hyperparameter

**NLP Precedent:** Chen et al. (GMSG, 2015) demonstrated dynamic sense component adjustment during training. GMSG adds/removes Gaussian mixture components per word based on training signal — the word-level version of what SGS does at the vocabulary level.

**Falsifiable Predictions:**
1. Models with adaptive density converge ≥ 10% faster (fewer steps to same loss) than fixed-vocabulary
2. Polysemous words (e.g., "bank", "spring", "bat") will have more Gaussians after training than monosemous words
3. If adaptive density shows no improvement, fixed vocabulary with pre-assigned polysemy counts works equally well

**Evidence:** 3DGS (Kerbl et al., 2023) validates in vision. GMSG (Chen et al., 2015) validates at word level in NLP.

**Assurance Level:** L1 (validated in adjacent domains; SGS-specific validation needed)

---

## Phase 2: Logical Verification

| Atom | Constraint Check | Type Check | Consistency | Verdict |
|---|---|---|---|---|
| A1 (Primitive) | Cholesky ensures PSD ✓; sigmoid ensures α∈[0,1] ✓ | Dimensions consistent across spaces ✓ | Compatible with A2-A7 ✓ | **PASS → L1** |
| A2 (Kernel) | Temperature τ prevents underflow at d_s=64 ✓ (needs empirical check) | Input/output types correct ✓ | Feeds into A3 correctly ✓ | **PASS → L1** (pending Phase 0) |
| A3 (Compositing) | Weights bounded ∈ [0,1] ✓; transmittance monotonic ✓ | Output ∈ ℝ^d_f ✓ | Gradient flow proven ✓ | **PASS → L1** (composition quality untested) |
| A4 (Viewpoint) | P·Σ·P^T preserves PSD ✓; projection well-defined ✓ | Multi-head concatenation dimensions match ✓ | Compatible with A3 rendering ✓ | **PASS → L1** |
| A5 (Multi-Pass) | Residual connections prevent gradient vanishing ✓; parameter counts feasible ✓ | MLP input/output dimensions consistent ✓ | Each pass uses A3; updates feed A2 ✓ | **PASS → L1** |
| A6 (Operators) | Soft-type is differentiable ✓; stack introduces sequential dependency ⚠ | Type probabilities sum to 1 ✓ | Breaks parallelism assumption from A3 ⚠ | **REFINE** — stack limits GPU parallelism |
| A7 (Adaptive) | Thresholds are hyperparameters, not learned ⚠; opacity reset is aggressive | Split operation well-defined ✓ | Must re-index vocabulary after split/prune ⚠ | **PASS → L1** (implementation complexity noted) |

**A6 Refinement:** The scope stack mechanism introduces sequential dependency that limits GPU parallelism within a rendering pass. Resolution options:
- (a) Limit stack depth to 2-3 (covers most nesting in natural language)
- (b) Replace stack with learned positional scope markers in the feature vector
- (c) Handle scope entirely in the multi-pass mechanism (pass 1: flat; pass 2+: scopeaware via updated features)

**Recommendation:** Option (c) — defer scope handling to multi-pass updates. This keeps each individual rendering pass fully parallel and uses the FFN in Atom A5 to learn scope-sensitive features.

---

## Phase 3: Evidence Validation

| Atom | Evidence Source | Confidence Level (CL) | Notes |
|---|---|---|---|
| A1 | Vilnis (2015), Athiwaratkun (2017, 2018), Yuksel (2021), Huang (2026) | CL=3 (multiple replications) | Gaussian primitives work for words — validated across 7 papers over 11 years |
| A2 | Mathematical analysis + Phase 0 experiment (pending) | CL=2 (theoretical + planned) | Temperature scaling is standard in ML; d_s=64 regime needs empirical confirmation |
| A3 | No direct precedent | CL=1 (theoretical only) | **WEAKEST LINK.** The entire approach depends on this untested atom |
| A4 | EWA Splatting (Zwicker, 2002); multi-head specialization (Clark, 2019) | CL=2 (validated in adjacent domain) | Projection is mathematically identical to 3DGS camera projection |
| A5 | Diffusion-LM (Li, 2022); NAR refinement (Lee, 2018); Kiruluta (2026) | CL=2 (validated analogies) | Iterative refinement is well-established; SGS-specific dynamics untested |
| A6 | No precedent | CL=0 (speculative) | **SECOND WEAKEST.** Soft-type operator mechanism has no empirical support |
| A7 | 3DGS (Kerbl, 2023); GMSG (Chen, 2015) | CL=2 (validated in adjacent domains) | Split/prune works for vision and word senses; vocabulary-level validation needed |

---

## Phase 4: Trust Audit

### R_eff Calculation

**R_eff = min(CL across dependency chain) × validity_discount**

| Atom | Direct CL | Dependencies | Chain CL | R_eff |
|---|---|---|---|---|
| A1 (Primitive) | 3 | None | 3 | **0.90** |
| A2 (Kernel) | 2 | A1 | 2 | **0.70** |
| A3 (Compositing) | 1 | A1, A2 | 1 | **0.35** ← WLNK |
| A4 (Viewpoint) | 2 | A1, A2, A3 | 1 | **0.35** |
| A5 (Multi-Pass) | 2 | A1, A2, A3, A4 | 1 | **0.35** |
| A6 (Operators) | 0 | A1, A2, A3 | 0 | **0.10** ← WEAKEST |
| A7 (Adaptive) | 2 | A1 | 2 | **0.70** |

### Weakest Link Analysis (WLNK)

**WLNK = A3 (Compositing Equation)** for the core architecture
**WLNK = A6 (Operator Gaussians)** for non-monotonic language

The entire SGS system's credibility is bounded by A3 — whether alpha-compositing produces compositionally meaningful sentence representations. Every downstream atom (viewpoints, multi-pass, operators, generation) depends on this.

### Bias Audit

| Bias | Check | Result |
|---|---|---|
| **Pet Idea Bias** | Is the rendering analogy over-fitted? | ⚠ Yes — the analogy is rhetorically powerful but A3 (the core novel claim) has CL=1. The aesthetic appeal of "meaning is rendering" may inflate confidence. |
| **Not Invented Here** | Are we ignoring simpler alternatives? | ⚠ Partially — Gaussian Transformer (A1 hybrid) and Gaussian Mixture Attention (A3 hybrid) may achieve 80% of the benefits with 20% of the risk. |
| **Sunk Cost** | Is the literature review biasing toward validation? | ✓ No — the orthogonal challenge and honest assessments throughout address this. |

---

## Phase 5: Decision — Atomic Component Prioritization

### Component Dependency Graph

```
A1 (Primitive)
├── A2 (Kernel)
│   └── A3 (Compositing) ← WLNK: test FIRST
│       ├── A4 (Viewpoint)
│       ├── A5 (Multi-Pass)
│       └── A6 (Operators) ← WEAKEST: test LAST
└── A7 (Adaptive Density) [independent of A3]
```

### Experimental Priority Order

| Priority | Atom | Experiment | Timeline | Gate |
|---|---|---|---|---|
| **P0** | A2 | Numerical feasibility at d_s=64 | Week 1-2 | If fails → reduce d_s to 32 or try Cauchy kernel |
| **P1** | A3 | Compositing vs. mean-pooling on STS-B | Month 1-3 | **KILL GATE.** If fails → pivot to hybrid (Gaussian Transformer). SGS as proposed is dead. |
| **P2** | A5 | Multi-pass ablation (P=1 vs P=4 vs P=8) | Month 2-3 | If P=1 = P=8 → single-pass suffices; multi-pass is overhead |
| **P3** | A4 | Multi-view specialization probe | Month 3-4 | If heads don't specialize → single-view is sufficient |
| **P4** | A7 | Adaptive density vs. fixed vocabulary | Month 4-6 | If no improvement → fixed vocabulary is simpler |
| **P5** | A6 | Operator Gaussians on negation/quantification | Month 5-7 | If fails → handle logic in FFN within multi-pass |

### The Kill Gate

**A3 is the kill gate.** If Phase 1 experiments show that alpha-compositing sentence representations do NOT outperform mean-pooling of Gaussian means on STS-B (target: ≥ 0.78 Spearman), then:

1. The rendering equation adds nothing to the Gaussian representation
2. The 3DGS analogy fails at the composition level
3. SGS should pivot to **Gaussian Transformer** (A1 hybrid) — keeping the Gaussian primitive but using standard attention for composition

This is not a failure of the research program — it narrows the contribution to "Gaussians as primitives" (validated) rather than "rendering as composition" (falsified).

---

## Summary: The Seven Atoms of SGS

| # | Atom | Definition | Assurance | Priority |
|---|---|---|---|---|
| A1 | Semantic Gaussian | G = (μ, L, α, f, t) in dual-space (d_s=64 splatting + d_f=512 features) | **L2** | Foundation |
| A2 | Gaussian Kernel | K(q,μ,Σ) = exp(-D_M/2τ) with learned temperature τ | **L1** | P0 (test first) |
| A3 | Compositing Equation | Meaning = Σ wᵢ·fᵢ with transmittance-gated weights, sequence-ordered | **L0** | **P1 (KILL GATE)** |
| A4 | Semantic Viewpoint | V = (P, q): projection + query, multi-view via heads | **L1** | P3 |
| A5 | Multi-Pass Rendering | P passes of render→update→FFN with residual connections | **L1** | P2 |
| A6 | Operator Gaussians | Soft-typed operators for negation/quantification/scope | **L0** | P5 (last) |
| A7 | Adaptive Density | Split/prune/clone based on accumulated gradients | **L1** | P4 |

**System R_eff: 0.35** (bounded by A3, the untested compositing equation)

**To reach R_eff ≥ 0.70:** Validate A3 empirically (Phase 1, STS-B ≥ 0.78).
