# SGS Mathematical Claims Requiring Formal Proof

**Purpose:** This document extracts every mathematical assertion made in the SGS whitepaper and atomic specification that requires formal verification. Each claim is stated precisely with its assumptions, the specific property to be proved (or disproved), and its criticality to the overall architecture.

**Intended recipient:** Aristotle AI (or equivalent mathematical proof assistant)
**Source documents:** `v4_literature_validated.md`, `fpf_atomic_specification.md`

---

## How to Read This Document

Each claim follows this structure:
- **Statement**: The precise mathematical claim
- **Assumptions**: What must hold for the claim to be meaningful
- **To prove**: The specific property or inequality
- **Criticality**: How much of SGS depends on this claim (CRITICAL / HIGH / MEDIUM / LOW)
- **Known results**: Any existing theorems that may help
- **Counterexample to check**: What would disprove it

---

## SECTION 1: Covariance and Parameterization (Atom A1)

### Claim 1.1: Cholesky Factorization Guarantees PSD

**Statement:** For any lower-triangular matrix L ∈ ℝ^(d×d) with positive diagonal entries and any ε > 0, the matrix Σ = LL^T + εI is symmetric positive definite.

**Assumptions:**
- L is lower-triangular: L_ij = 0 for j > i
- diag(L) > 0 (enforced by softplus activation on diagonal)
- ε > 0 (numerical jitter, fixed at 1e-6)

**To prove:**
1. Σ is symmetric: Σ^T = Σ
2. Σ is positive definite: x^T Σ x > 0 for all x ≠ 0
3. Σ remains PD under gradient updates to L (i.e., PD is preserved throughout training)

**Criticality:** HIGH — if Σ is not PSD, the Mahalanobis distance and Gaussian kernel are undefined.

**Known results:** Standard result in linear algebra. LL^T is PSD for any L; adding εI with ε > 0 makes it PD. The proof is straightforward but should be stated for completeness.

**Counterexample to check:** Can gradient updates make diag(L) ≤ 0 if softplus is not applied? (Yes — hence the softplus requirement on the diagonal.)

---

### Claim 1.2: Parameter Count of Lower-Triangular L

**Statement:** A lower-triangular matrix L ∈ ℝ^(d×d) has exactly d(d+1)/2 free parameters.

**Assumptions:** d = d_s (splatting space dimension)

**To prove:** |{(i,j) : 1 ≤ j ≤ i ≤ d}| = d(d+1)/2

**Criticality:** LOW — bookkeeping, but affects model size calculations.

**At d_s = 64:** 64 × 65 / 2 = 2,080 parameters per Gaussian. Verify.

---

## SECTION 2: Gaussian Kernel Properties (Atom A2)

### Claim 2.1: Temperature-Scaled Kernel Is Well-Defined

**Statement:** For any q, μ ∈ ℝ^d, any SPD matrix Σ ∈ ℝ^(d×d), and any τ > 0:

```
K(q, μ, Σ) = exp(-½ · (q - μ)^T Σ^{-1} (q - μ) / τ)
```

satisfies: K ∈ (0, 1], K is smooth, and K is a valid positive-definite kernel.

**Assumptions:**
- Σ is SPD (guaranteed by Claim 1.1)
- τ > 0

**To prove:**
1. K(q, μ, Σ) ∈ (0, 1] for all q, μ, Σ, τ
2. K(μ, μ, Σ) = 1 (maximum at center)
3. K is C^∞ (infinitely differentiable) with respect to q, μ, and the entries of L (via Σ = LL^T)
4. K is a positive-definite kernel in the Mercer sense: for any set of points {q_1, ..., q_n}, the kernel matrix K_ij = K(q_i, q_j, Σ) is PSD

**Criticality:** HIGH — kernel must be valid for the rendering equation to define a proper weighted sum, and for gradient-based optimization to work.

**Known results:** The Gaussian (RBF) kernel is a well-known Mercer kernel (Scholkopf & Smola, 2002). The temperature parameter τ rescales the bandwidth. The key question is whether the *anisotropic* form (with full Σ rather than scalar σ²I) preserves positive-definiteness.

**Counterexample to check:** Does anisotropic Σ preserve the Mercer property? (Yes — the anisotropic Gaussian kernel exp(-½ (x-y)^T M (x-y)) is PD for any PD matrix M, by composition with the isotropic Gaussian in a Mahalanobis-transformed space.)

---

### Claim 2.2: Expected Mahalanobis Distance

**Statement:** If q ~ N(μ, Σ), then E[(q - μ)^T Σ^{-1} (q - μ)] = d.

**Assumptions:**
- q is drawn from N(μ, Σ)
- Σ is SPD

**To prove:** E[D_M] = d, where D_M = (q - μ)^T Σ^{-1} (q - μ)

**Criticality:** HIGH — this determines the correct temperature setting. If E[D_M] = d, then τ = d normalizes the kernel so E[K] = exp(-0.5) ≈ 0.607.

**Known results:** D_M follows a chi-squared distribution with d degrees of freedom when q ~ N(μ, Σ). E[χ²_d] = d. Standard result.

**Additional question:** What is Var[D_M]? Answer: Var[χ²_d] = 2d. At d=64, std(D_M) = √128 ≈ 11.3, so D_M ∈ [64 ± 22.6] with ~95% probability. This means K ∈ [exp(-0.68), exp(-0.32)] ≈ [0.51, 0.73] — a narrow, well-behaved range. Verify.

---

### Claim 2.3: Concentration of Mahalanobis Distance in High Dimensions

**Statement:** As d → ∞, for q drawn from N(μ, Σ), the Mahalanobis distance D_M concentrates around d:

```
P(|D_M - d| > t√(2d)) ≤ 2exp(-t²/2)    (sub-Gaussian tail)
```

**Assumptions:** q ~ N(μ, Σ), Σ SPD

**To prove:** The relative deviation |D_M - d|/d → 0 in probability as d → ∞, with the stated tail bound.

**Criticality:** MEDIUM — determines how "peaked" the kernel is at d=64. If concentration is too tight, all points at distance ~d look identical (killing discriminative power). If too loose, the kernel is too variable (unstable training).

**At d = 64:** Relative std = √(2/d) = √(1/32) ≈ 0.177 = 17.7%. This is moderate — enough spread for discrimination but concentrated enough for stability.

**Known results:** Chi-squared concentration inequalities (Laurent & Massart, 2000).

---

### Claim 2.4: Sparsity of Kernel Evaluations

**Statement:** For a trained SGS model with N Gaussians and a query point q, define:

```
S(q) = |{i : K(q, μ_i, Σ_i) > ε_sparse}|
```

The claim is that S(q) << N for most queries (effective sparsity).

**Assumptions:**
- ε_sparse = 1e-3
- τ = d_s = 64
- Gaussian means μ_i are distributed across semantic space (not all co-located)

**To prove (or estimate):** Under what conditions on the distribution of μ_i does S(q) = O(k) where k << N?

**Specifically:** If Gaussians are uniformly distributed in a d-dimensional ball of radius R, and each has isotropic covariance σ²I, what fraction have K > ε_sparse for a random query?

```
K > ε_sparse
⟺ D_M / τ < -2 ln(ε_sparse)
⟺ ||q - μ_i||² / (σ² · τ) < -2 ln(ε_sparse)
⟺ ||q - μ_i|| < σ · √(τ · (-2 ln(ε_sparse)))
```

At τ = 64, ε_sparse = 1e-3: radius = σ · √(64 · 6.91) = σ · √442 ≈ 21σ

The fraction of Gaussians within radius 21σ in a d=64 ball depends on the volume ratio, which goes as (21σ/R)^d. For this to be small, R >> 21σ.

**Criticality:** CRITICAL — the entire efficiency claim (O(n·k) instead of O(n²)) depends on this. If sparsity doesn't hold, SGS is slower than transformers.

**Counterexample to check:** In high dimensions, if Gaussians cluster tightly (semantic space is compact), S(q) ≈ N for all queries — no sparsity. This is the concentration-of-measure failure mode.

**What needs proof:** A bound on S(q) as a function of d, N, the spread of Gaussian means, and the average Gaussian scale.

---

### Claim 2.5: Gradient of the Kernel

**Statement:** The gradients of K with respect to the Gaussian parameters are:

```
∂K/∂μ = K · Σ^{-1}(q - μ) / τ

∂K/∂L = K · [Σ^{-1}(q-μ)(q-μ)^T Σ^{-1} - Σ^{-1}] · L / (2τ)
         (via chain rule through Σ = LL^T)

∂K/∂τ = K · D_M / (2τ²)
```

**Assumptions:** Σ = LL^T + εI, K as defined in Claim 2.1

**To prove:** Derive each gradient explicitly and verify:
1. ∂K/∂μ points FROM μ TOWARD q (gradient pulls mean toward query) — confirms correct learning dynamics
2. ∂K/∂L adjusts covariance to better fit the query-mean distance
3. No gradient is identically zero for non-degenerate configurations
4. Gradients are bounded when K is bounded away from 0

**Criticality:** HIGH — gradient correctness is essential for training. If gradients vanish or explode, the model won't learn.

**Known results:** Standard multivariate calculus. The key subtlety is the chain rule through the Cholesky factor L → Σ → K.

---

## SECTION 3: Compositing Equation Properties (Atom A3)

### Claim 3.1: Blending Weights Sum to At Most 1

**Statement:** For the rendering equation with weights wᵢ = αᵢ · Kᵢ · Tᵢ and transmittance Tᵢ = ∏_{j<i} (1 - αⱼ · Kⱼ):

```
Σᵢ₌₁ⁿ wᵢ = 1 - Tₙ₊₁ ≤ 1
```

**Assumptions:**
- αᵢ ∈ [0, 1] for all i
- Kᵢ ∈ [0, 1] for all i
- Therefore αᵢ · Kᵢ ∈ [0, 1]

**To prove:**
1. Σᵢ wᵢ = 1 - ∏ᵢ₌₁ⁿ (1 - αᵢKᵢ) (telescoping product identity)
2. Therefore 0 ≤ Σᵢ wᵢ ≤ 1
3. Equality Σᵢ wᵢ = 1 holds iff Tₙ₊₁ = 0 (full opacity — all transmittance absorbed)

**Criticality:** HIGH — ensures the output Meaning(q) is a convex-like combination of features (bounded, interpretable weights).

**Known results:** Standard result in volume rendering / alpha compositing (Porter & Duff, 1984; Max, 1995). The "over" operator has this telescoping property.

**Proof sketch:** Define aᵢ = αᵢKᵢ. Then T₁ = 1, Tᵢ₊₁ = Tᵢ(1-aᵢ). So wᵢ = aᵢTᵢ and:
Σᵢ wᵢ = Σᵢ aᵢTᵢ = Σᵢ (Tᵢ - Tᵢ₊₁) = T₁ - Tₙ₊₁ = 1 - Tₙ₊₁. ∎

**But verify:** Does this hold when αᵢ and Kᵢ are functions of learnable parameters? (Yes — the algebra depends only on aᵢ ∈ [0,1], not on how aᵢ is parameterized.)

---

### Claim 3.2: Monotonic Transmittance

**Statement:** Tᵢ is non-increasing: T₁ ≥ T₂ ≥ ... ≥ Tₙ ≥ Tₙ₊₁ ≥ 0.

**Assumptions:** αᵢKᵢ ∈ [0, 1] for all i

**To prove:**
- Tᵢ₊₁ = Tᵢ · (1 - αᵢKᵢ) ≤ Tᵢ since (1 - αᵢKᵢ) ∈ [0, 1]
- Tᵢ ≥ 0 for all i (product of non-negative terms)

**Criticality:** MEDIUM — ensures earlier tokens can only have MORE influence than later tokens (before multi-pass correction).

---

### Claim 3.3: Complete Gradient Flow Through Compositing

**Statement:** For every parameter θ ∈ {μᵢ, Lᵢ, αᵢ, fᵢ} of every Gaussian Gᵢ with wᵢ > 0, the gradient ∂Meaning(q)/∂θ ≠ 0.

**Assumptions:**
- wᵢ > 0 (Gaussian contributes non-trivially)
- Meaning(q) = Σⱼ wⱼfⱼ as defined in Atom A3

**To prove:** For each parameter type:

1. **∂Meaning/∂fᵢ = wᵢ · I** — non-zero whenever wᵢ > 0 ✓
2. **∂Meaning/∂αᵢ** — has two terms:
   - Direct: (∂wᵢ/∂αᵢ) · fᵢ = Kᵢ · Tᵢ · fᵢ
   - Indirect via transmittance: Σⱼ>ᵢ (∂wⱼ/∂αᵢ) · fⱼ
   - Prove: at least one of these is non-zero when wᵢ > 0
3. **∂Meaning/∂μᵢ** — via ∂Kᵢ/∂μᵢ, which propagates through both wᵢ and Tⱼ for j > i
   - Prove: ∂Kᵢ/∂μᵢ ≠ 0 when q ≠ μᵢ
4. **∂Meaning/∂Lᵢ** — via ∂Kᵢ/∂Lᵢ (from Claim 2.5)
   - Prove: ∂Kᵢ/∂Lᵢ ≠ 0 when q ≠ μᵢ

**Critical edge case:** When q = μᵢ, Kᵢ = 1 and ∂Kᵢ/∂μᵢ = 0 (at the peak of the Gaussian, the gradient of the kernel is zero). Does this cause gradient starvation for μᵢ?

**To prove or disprove:** For a query set Q = {q₁, ..., q_m}, if all queries equal μᵢ, then μᵢ receives zero positional gradient. But for a diverse query set (as in language modeling), the probability that all queries exactly equal any single μᵢ is zero (measure-zero event in continuous space).

**Criticality:** CRITICAL — gradient flow to all parameters is necessary for end-to-end training.

---

### Claim 3.4: Order-Dependence Captures Word Order

**Statement:** For two Gaussian scenes Ω₁ = (G_a, G_b) and Ω₂ = (G_b, G_a) differing only in order, Meaning_Ω₁(q) ≠ Meaning_Ω₂(q) in general.

**Assumptions:**
- G_a ≠ G_b (different Gaussians)
- Both have non-zero blending weights at q

**To prove:** Construct a concrete example where reordering changes the output. Specifically:

```
Ω₁: w_a = α_a · K_a · 1,  w_b = α_b · K_b · (1 - α_a · K_a)
Ω₂: w_b' = α_b · K_b · 1, w_a' = α_a · K_a · (1 - α_b · K_b)

Meaning_Ω₁ = w_a · f_a + w_b · f_b
Meaning_Ω₂ = w_b' · f_b + w_a' · f_a
```

These differ when α_a·K_a ≠ α_b·K_b (i.e., the two Gaussians have different effective opacities at q).

**Criticality:** MEDIUM — confirms the architecture captures syntactic word order, not just bag-of-words.

**Counterexample to check:** When α_a·K_a = α_b·K_b, ordering doesn't matter — the compositing is commutative for equally-weighted Gaussians. This is a degenerate case.

---

### Claim 3.5: The Rendering Equation Is a Special Case of Attention

**Statement:** There exists a mapping from SGS rendering parameters to attention parameters such that the rendering equation becomes a special case of (a variant of) the attention mechanism.

**Specifically:**
- Volume rendering: C = Σᵢ Tᵢ αᵢ Kᵢ · fᵢ
- Attention: y = Σᵢ softmax(sᵢ) · vᵢ

**To prove or disprove:** Can we express Tᵢ αᵢ Kᵢ as a softmax-like normalization of some score? Or conversely, can softmax attention be expressed as a special case of transmittance-gated compositing?

**Key difference:** Softmax normalizes: Σᵢ softmax(sᵢ) = 1 always. Transmittance-gated weights sum to 1 - Tₙ₊₁, which is ≤ 1 and can be < 1 (residual transmittance = "unaccounted meaning").

**Conjecture:** The two are NOT equivalent but are both instances of a more general "weighted aggregation" framework:

```
y = Σᵢ w(sᵢ, s₁..sₙ) · vᵢ
```

where w is:
- For attention: w(sᵢ, s₁..sₙ) = exp(sᵢ) / Σⱼ exp(sⱼ)  [global normalization]
- For rendering: w(sᵢ, s₁..sᵢ₋₁) = sᵢ · ∏ⱼ<ᵢ (1 - sⱼ)  [sequential transmittance]

**To prove:** That these are the only two natural solutions to some set of axioms (e.g., non-negativity, bounded sum, differentiability), or that there exist other valid weighting schemes.

**Criticality:** MEDIUM — theoretically interesting but not required for SGS to work. Would be a novel theoretical contribution if proven.

**Known results:** Ramsauer et al. (2021) proved attention = Hopfield energy minimization. Katharopoulos et al. (2020) showed attention is a kernel function. No paper connects volume rendering to either.

---

## SECTION 4: Viewpoint Projection Properties (Atom A4)

### Claim 4.1: Projected Covariance Preserves PSD

**Statement:** If Σ is SPD and P ∈ ℝ^(m×d) has rank m (full row rank), then Σ' = PΣP^T is SPD.

**Assumptions:**
- Σ ∈ ℝ^(d×d) is SPD
- P ∈ ℝ^(m×d) with m ≤ d and rank(P) = m

**To prove:** For all x ∈ ℝ^m, x ≠ 0: x^T PΣP^T x > 0.

**Proof sketch:** Let y = P^T x. Since P has full row rank, P^T has full column rank, so P^T x ≠ 0 when x ≠ 0. Then x^T PΣP^T x = y^T Σ y > 0 since Σ is SPD and y ≠ 0. ∎

**Criticality:** HIGH — projected Gaussians must remain valid for the rendering equation.

**Edge case:** If P does NOT have full row rank (degenerate projection), Σ' is PSD but not PD. The kernel K' is still defined but may have zero eigenvalues (meaning some directions in the projected space are infinitely uncertain). Should be avoided by regularization or architecture constraints.

---

### Claim 4.2: Multi-View Rendering Recovers More Information Than Single-View

**Statement:** For H > 1 viewpoints {V₁, ..., V_H} with distinct projections {P₁, ..., P_H}, the concatenated output:

```
MultiView = [Meaning₁(q₁); ...; Meaning_H(q_H)]  ∈ ℝ^(H·d_f)
```

captures strictly more information about the Gaussian scene Ω than any single viewpoint.

**To prove:** There exist Gaussian scenes Ω₁ ≠ Ω₂ such that Meaning_h(q_h; Ω₁) = Meaning_h(q_h; Ω₂) for some head h, but MultiView(Ω₁) ≠ MultiView(Ω₂).

**Criticality:** MEDIUM — justifies the multi-head architecture.

**Known results:** In 3DGS, this is trivially true — different camera angles see different parts of the scene. In semantic space, the question is whether semantic "occlusion" (transmittance depletion) creates analogous information hiding.

---

## SECTION 5: Multi-Pass Convergence (Atom A5)

### Claim 5.1: Multi-Pass Does Not Diverge

**Statement:** The iterative parameter update:
```
μᵢ^(p+1) = μᵢ^(p) + Δμᵢ^(p)
αᵢ^(p+1) = αᵢ^(p) · gate_i^(p)     where gate ∈ (0, 1]
fᵢ^(p+1) = fᵢ^(p) + FFN(fᵢ^(p), cᵢ^(p))
```

does not diverge (||μ^(p)||, ||f^(p)||, α^(p) remain bounded) for P passes.

**Assumptions:**
- MLP_μ has bounded output (e.g., via tanh activation on the final layer)
- gate ∈ (0, 1] (from sigmoid, ensures α is non-increasing)
- FFN has LayerNorm (standard transformer practice)

**To prove:**
1. ||μᵢ^(p)|| ≤ ||μᵢ^(0)|| + P · max_Δμ (bounded if Δμ is bounded)
2. αᵢ^(P) ≤ αᵢ^(0) (opacity can only decrease across passes)
3. ||fᵢ^(p)|| is controlled by LayerNorm

**Criticality:** HIGH — unbounded parameters during multi-pass would cause numerical explosion.

**Counterexample to check:** Without bounded Δμ (e.g., linear MLP with no activation), ||μ|| can grow exponentially with P. Tanh or clipping on Δμ is essential.

---

### Claim 5.2: Opacity Monotonicity Across Passes

**Statement:** If gate_i^(p) = σ(MLP_α(...)) ∈ (0, 1), then α_i^(P) ≤ α_i^(0) and specifically:

```
α_i^(P) = α_i^(0) · ∏_{p=0}^{P-1} gate_i^(p)
```

**To prove:** Since each gate ∈ (0, 1), the product is strictly decreasing.

**Implication:** Across passes, Gaussians can only become LESS salient, never more. This means disambiguation works by *suppressing* wrong senses, not amplifying correct ones.

**Criticality:** MEDIUM — understanding the multi-pass dynamics is important for architecture design. If we want amplification too, the gate should be ∈ (0, 2) or use a different parameterization.

**Design question (not a proof, but for Aristotle's consideration):** Is suppress-only sufficient for disambiguation, or does the model also need amplification? Amplification would require gate > 1, which changes the convergence properties.

---

## SECTION 6: Operator Gaussians (Atom A6)

### Claim 6.1: Soft-Type Maintains Differentiability

**Statement:** The contribution:
```
contribution_i = Σ_t p_t · effect_t(w_i, f_i, f_{next}, ...)
```

where p_t = softmax(W_type · f_i)_t and each effect_t is differentiable, is differentiable with respect to all parameters.

**Assumptions:**
- softmax is differentiable
- Each effect (content, negate, quantify, scope) is differentiable
- For negation: effect_negate = w_i · (-f_{next}) — depends on f_{i+1}

**To prove:** The composite function is differentiable (standard chain rule through softmax + linear combination of differentiable effects).

**Criticality:** LOW — straightforward from composition of differentiable functions. The real question is whether the gradient signal is useful, not whether it exists.

---

### Claim 6.2: Negation Via Sign Flip Is Learnable

**Statement:** A model with operator Gaussians can learn to represent "not X" by having the negation operator produce -f_X, which in the feature space corresponds to semantic negation.

**To prove or disprove:** In a trained word embedding space, is -f_X a good approximation of "not X"? Specifically:

1. Is f_"happy" + (-f_"happy") ≈ f_"neutral"? (Probably not — negation is not zero)
2. Is -f_"happy" ≈ f_"unhappy"? (Known to be approximately true for antonym pairs in Word2Vec)
3. More generally, does there exist a linear transformation N such that N·f_X better captures negation than -f_X?

**Criticality:** MEDIUM — if simple sign flip doesn't capture negation, a learned negation MLP is needed (adding parameters and complexity).

**Known results:** Mikolov et al. (2013) showed antonym relationships have consistent vector offsets in Word2Vec, but these are NOT simply sign flips. Negation in embedding spaces is more complex than scalar negation.

**Honest assessment:** Simple sign flip is likely insufficient. A learned negation transformation N(f) is more realistic but departs further from the pure splatting analogy.

---

## SECTION 7: Adaptive Density Control (Atom A7)

### Claim 7.1: Split Operation Preserves Local Density

**Statement:** When Gaussian G with parameters (μ, Σ, α) is split into G_a and G_b with:
```
μ_a = μ + ε·v_max,  μ_b = μ - ε·v_max
Σ_a = Σ_b = Σ/4
α_a = α_b = α
```

the total "mass" (integrated density) of G_a + G_b approximates that of G for small ε.

**To prove:** The integral ∫ (α·N(x|μ_a, Σ/4) + α·N(x|μ_b, Σ/4)) dx ≈ ∫ 2α·N(x|μ, Σ) dx for ε → 0.

**Note:** The factor of 2 means the split increases total mass by 2x. The original 3DGS paper addresses this by also halving opacity: α_a = α_b = α/2. Should SGS do the same?

**Criticality:** MEDIUM — incorrect mass preservation causes density drift during training.

**Counterexample to check:** If we don't halve opacity, each split doubles the effective contribution of that semantic region, biasing the model toward over-represented areas.

---

## SUMMARY: Priority Ordering for Proofs

| Priority | Claim | Criticality | Difficulty | Status |
|---|---|---|---|---|
| **1** | 3.1 (Weights sum ≤ 1) | HIGH | Easy (telescoping) | Prove |
| **2** | 3.3 (Complete gradient flow) | CRITICAL | Medium | Prove, especially q = μ edge case |
| **3** | 2.2 + 2.3 (Mahalanobis distribution at d=64) | HIGH | Easy (chi-squared) | Verify numerics |
| **4** | 2.4 (Sparsity bound) | CRITICAL | Hard | Derive bound as f(d, N, σ, R) |
| **5** | 3.5 (Rendering ↔ Attention) | MEDIUM | Hard (novel) | Prove or characterize relationship |
| **6** | 5.1 (Multi-pass stability) | HIGH | Medium | Prove bounded with tanh |
| **7** | 1.1 (Cholesky PSD) | HIGH | Easy (standard) | State for completeness |
| **8** | 4.1 (Projected covariance PSD) | HIGH | Easy (standard) | State for completeness |
| **9** | 2.1 (Kernel properties) | HIGH | Easy (known) | Verify anisotropic Mercer |
| **10** | 6.2 (Negation learnability) | MEDIUM | Hard (empirical) | Theoretical analysis + conjecture |
| **11** | 3.4 (Order-dependence) | MEDIUM | Easy (constructive) | Construct example |
| **12** | 5.2 (Opacity monotonicity) | MEDIUM | Easy | Direct computation |
| **13** | 7.1 (Split mass preservation) | MEDIUM | Medium | Integral calculation |
| **14** | 4.2 (Multi-view information gain) | MEDIUM | Medium | Constructive proof |
| **15** | 1.2 (Parameter count) | LOW | Trivial | Verify |
| **16** | 6.1 (Soft-type differentiability) | LOW | Easy | Chain rule |

---

## OPEN QUESTIONS (Not Claims — For Mathematical Exploration)

### OQ1: Expressiveness of Transmittance-Gated Composition

Can the rendering equation approximate any continuous function f: (ℝ^d_f)^n → ℝ^d_f given enough Gaussians? (Universal approximation property)

Specifically: for any target function F mapping n feature vectors to an output, and any ε > 0, does there exist a Gaussian scene Ω with parameters such that ||Meaning(q; Ω) - F(f₁, ..., fₙ)|| < ε?

If YES: the architecture is universal (at least for single-pass rendering).
If NO: characterize what functions CAN be expressed — this defines the architectural bias.

### OQ2: Optimal Temperature Schedule

Is there an optimal τ* that maximizes the mutual information I(Meaning(q); Ω) given a Gaussian scene? How does τ* relate to d_s and the distribution of Gaussians?

### OQ3: Information-Theoretic Capacity

How many bits of information can a single rendering pass extract from a Gaussian scene of N Gaussians? Is there an analog of the data processing inequality for the rendering equation?

### OQ4: Relationship to Optimal Transport

Is the SGS rendering equation related to the Wasserstein distance between the query distribution and the Gaussian scene? Can the rendered meaning be interpreted as the barycentric projection of the query onto the scene in Wasserstein space?
