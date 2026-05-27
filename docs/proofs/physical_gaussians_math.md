# Physical Gaussians: Formal Mathematical Specification

**Purpose:** Define the Physical Gaussian primitive with mathematical precision.
State all claims that require formal verification. Structure claims for
Aristotle (Lean 4 formal prover).

**Source:** `docs/papers/physical_gaussians.md`, `docs/papers/physical_gaussians_literature_review.md`

---

## 1. Definitions

### Definition 1.1: Physical Gaussian Primitive

A Physical Gaussian is a tuple:

```
G = (p, S, q, α, c, e_s, e_p, ℓ)
```

where:
- `p ∈ ℝ³` — position (center of the Gaussian in world space)
- `S ∈ ℝ³` — log-scale (anisotropic, S_i = log(σ_i))
- `q ∈ ℍ₁` — unit quaternion (rotation, ||q|| = 1)
- `α ∈ ℝ` — opacity logit (actual opacity = σ(α) ∈ (0,1))
- `c ∈ ℝ³` — color (RGB in [0,1], or SH DC coefficients)
- `e_s ∈ ℝ^{d_s}` — semantic embedding (d_s = 128-300)
- `e_p ∈ ℝ^{d_p}` — physical embedding (d_p = 32-64)
- `ℓ ∈ Σ*` — label (finite string from alphabet Σ)

### Definition 1.2: Covariance Matrix

The 3D covariance matrix of a Physical Gaussian is:

```
Σ = R(q) · diag(exp(2·S)) · R(q)^T
```

where `R(q) : ℍ₁ → SO(3)` is the quaternion-to-rotation-matrix map:

```
R(w,x,y,z) = | 1-2(y²+z²)   2(xy-wz)     2(xz+wy)   |
              | 2(xy+wz)     1-2(x²+z²)   2(yz-wx)   |
              | 2(xz-wy)     2(yz+wx)     1-2(x²+y²) |
```

### Definition 1.3: Material Similarity

For two Physical Gaussians G_i, G_j, the material similarity is:

```
M(G_i, G_j) = w_s · cos(e_s^i, e_s^j)
             + w_g · exp(-||S_i - S_j||₂)
             + w_α · exp(-|α_i - α_j|)
             + w_d · exp(-||p_i - p_j||₂ / r)
             + w_Σ · Φ(Σ_i, Σ_j)
```

where:
- `cos(a,b) = ⟨a,b⟩ / (||a||·||b||)` is cosine similarity
- `r > 0` is a spatial scale parameter
- `w_s, w_g, w_α, w_d, w_Σ > 0` are weights summing to 1
- `Φ(Σ_i, Σ_j)` is a covariance similarity function (defined below)

### Definition 1.4: Covariance Similarity

```
Φ(A, B) = exp(-½ · ||log(λ(A)) - log(λ(B))||₂)
```

where `λ(A) ∈ ℝ³` are the eigenvalues of A (guaranteed positive since A is SPD).

### Definition 1.5: Physics Prediction Function

```
f_phys : ℝ^{d_s} × ℝ³ × ℝ × ℝ⁶ → ℝ^{d_p}
f_phys(e_s, S, α, v(Σ)) = MLP(e_s ⊕ S ⊕ α ⊕ v(Σ))
```

where `v(Σ) ∈ ℝ⁶` is the upper-triangular vectorization of Σ (6 unique entries of a symmetric 3x3 matrix), and `⊕` denotes concatenation.

### Definition 1.6: Material Region

A material region is a connected component of the graph:

```
G_M = (V, E)
V = {G_1, ..., G_N}  (all Gaussians in the scene)
E = {(G_i, G_j) : M(G_i, G_j) > τ}
```

where τ ∈ (0,1) is a threshold. Gaussians in the same connected component share physical behavior.

### Definition 1.7: Collapse Operators

The Physical Gaussian admits projection operators for each subsystem:

```
π_render(G) = (p, Σ, σ(α), c)           — visual rendering
π_physics(G) = (p, Σ, e_p)              — physics simulation
π_semantic(G) = (p, e_s, ℓ)             — scene understanding
π_audio(G) = (p, e_p, e_s, Σ)           — sound synthesis
```

Each operator discards the components irrelevant to its subsystem.

---

## 2. Claims Requiring Formal Proof

### Claim P1: Covariance is SPD (from rotation + scale)

**Statement:** For any unit quaternion q ∈ ℍ₁ and any S ∈ ℝ³, the matrix Σ = R(q) · diag(exp(2S)) · R(q)^T is symmetric positive definite.

**Assumptions:**
- q ∈ ℍ₁ (||q|| = 1), so R(q) ∈ SO(3) (orthogonal, det = 1)
- S ∈ ℝ³ (arbitrary real log-scales)

**To prove:**
1. Σ is symmetric: Σ^T = Σ
2. Σ is positive definite: x^T Σ x > 0 for all x ≠ 0

**Criticality:** CRITICAL — covariance must be SPD for the Gaussian to be well-defined and for covariance similarity (Claim P4) to be meaningful.

**Proof sketch:** R is orthogonal, D = diag(exp(2S)) has strictly positive diagonal, so RDR^T is congruent to D and inherits positive-definiteness.

---

### Claim P2: Material Similarity is a Valid Metric (Bounded)

**Statement:** For any two Physical Gaussians G_i, G_j:

```
0 ≤ M(G_i, G_j) ≤ 1
```

and M(G_i, G_i) = 1 (self-similarity is maximal).

**Assumptions:**
- Weights w_s, w_g, w_α, w_d, w_Σ ∈ [0,1] with Σ w_k = 1
- Cosine similarity ∈ [-1, 1] (need w_s component to be ∈ [0,1], so we use (cos+1)/2)
- All exponential terms ∈ (0, 1]

**To prove:**
1. Each component is in [0, 1]
2. The weighted sum is in [0, 1]
3. Self-similarity: M(G, G) = 1

**Criticality:** HIGH — if M is unbounded or not self-maximal, the clustering algorithm (material regions) has undefined behavior.

**Adjustment needed:** Cosine similarity ranges [-1,1]. Redefine the semantic component as `(cos(e_s^i, e_s^j) + 1) / 2` to map to [0,1].

---

### Claim P3: Material Regions Form a Valid Partition (at threshold τ)

**Statement:** For any threshold τ ∈ (0,1), the material regions (connected components of G_M) form a partition of V: every Gaussian belongs to exactly one material region.

**Assumptions:**
- G_M is an undirected graph on V = {G_1, ..., G_N}
- E = {(i,j) : M(G_i, G_j) > τ}

**To prove:** The connected components of any undirected graph partition the vertex set (standard graph theory).

**Criticality:** MEDIUM — correctness of the segmentation algorithm.

**Note:** This is a standard result (connected components partition vertices). The non-trivial claim is that the resulting partition is *physically meaningful*, which requires empirical validation, not formal proof.

---

### Claim P4: Covariance Similarity is Well-Defined for SPD Matrices

**Statement:** For any two SPD matrices A, B ∈ S³₊₊:

```
Φ(A, B) = exp(-½ · ||log(λ(A)) - log(λ(B))||₂) ∈ (0, 1]
```

and Φ(A, A) = 1.

**Assumptions:**
- A, B are 3x3 symmetric positive definite
- λ(A), λ(B) are eigenvalues, all strictly positive

**To prove:**
1. log(λ(A)) is well-defined (eigenvalues > 0 for SPD)
2. Φ ∈ (0, 1] (exponential of non-positive quantity)
3. Φ(A, A) = 1 (||log λ - log λ|| = 0)
4. Φ is symmetric: Φ(A, B) = Φ(B, A)

**Criticality:** HIGH — feeds into material similarity metric.

---

### Claim P5: Physics Prediction Continuity

**Statement:** If f_phys is a feedforward neural network with continuous activations (ReLU, sigmoid, or tanh), then f_phys is continuous as a function ℝ^{d_s + 10} → ℝ^{d_p}.

**Assumptions:**
- f_phys = W_n · σ_{n-1} · ... · σ_1 · W_1 (composition of affine maps and elementwise activations)
- Each activation σ_k is continuous (ReLU, sigmoid, tanh, LayerNorm)

**To prove:** The composition of continuous functions is continuous.

**Criticality:** MEDIUM — ensures that small changes in semantic embedding produce small changes in predicted physics (no discontinuous material jumps).

**Note:** ReLU is continuous but not differentiable at 0. Continuity holds; Lipschitz continuity gives a stronger bound on the rate of change.

---

### Claim P6: Correlation Hypothesis (Mutual Information Bound)

**Statement:** The mutual information between the physical embedding and the (semantic, geometric) features is bounded below:

```
I(e_p ; e_s, S, α) ≥ I(e_p ; e_s) ≥ H(e_p) - H(e_p | material_class)
```

where material_class is the discrete material label derived from e_s.

**Assumptions:**
- e_p is determined (with noise) by the material class
- e_s encodes the material class (among other things)
- S, α provide additional discriminative signal

**To prove:** The data processing inequality gives the first ≥. The second ≥ is the bound from class-conditional entropy reduction.

**Criticality:** CRITICAL (THEORETICAL) — this is the core hypothesis. If semantic embeddings do NOT carry sufficient information about material properties, the prediction network cannot work. This is ultimately an empirical claim, but the information-theoretic bound provides a necessary condition.

**Empirical validation:** Train f_phys on ground-truth (material, e_p) pairs. If prediction R² > 0.7, the hypothesis holds practically.

---

### Claim P7: Collapse Operators are Projections

**Statement:** Each collapse operator π_X is an idempotent linear projection on the product space:

```
π_X ∘ π_X = π_X
```

and the operators are independent:

```
π_render(G) does not depend on e_p
π_physics(G) does not depend on c
π_semantic(G) does not depend on c, α (beyond what e_s encodes)
```

**Assumptions:**
- Operators are defined as coordinate projections on the product space

**To prove:** Coordinate projections are idempotent (trivial). The independence property is by construction (the operator reads only specific fields).

**Criticality:** LOW — definitional, but important for the conceptual framework (different subsystems see different "faces" of the same primitive).

---

### Claim P8: Densification Preserves Material Coherence

**Statement:** When a Physical Gaussian G is cloned or split into children G_a, G_b:

```
M(G_a, G_b) ≥ 1 - δ
```

where δ depends on the perturbation magnitude (random offset for clone, axis-split for split).

**Assumptions:**
- Clone: G_a has position p + ε (small random offset), same e_s, e_p, S, α
- Split: G_a, G_b have positions p ± offset along longest axis, scale reduced by factor β

**To prove:**
- For clone: M(G_a, G_b) → 1 as ε → 0 (by continuity of M in p)
- For split: M(G_a, G_b) = 1 - w_g·(1 - exp(-Δ_S)) - w_d·(1 - exp(-2·offset/r))

This provides a computable bound on how much material coherence degrades during densification.

**Criticality:** HIGH — if densification breaks material regions (children end up in different clusters), the physics becomes inconsistent.

---

### Claim P9: Physical Embedding Dimensionality Sufficiency

**Statement:** For K discrete material classes with distinct physical behavior, d_p ≥ ⌈log₂(K)⌉ dimensions suffice for lossless class separation, and d_p = O(K) dimensions suffice for continuous interpolation between all class pairs.

**Assumptions:**
- K material classes with distinct target vectors t_1, ..., t_K ∈ ℝ^{d_p}
- "Lossless separation" means min_{i≠j} ||t_i - t_j|| > 0

**To prove:**
1. K points in ℝ^d can be separated for d ≥ ⌈log₂(K)⌉ (basic dimension counting)
2. For smooth interpolation on a K-simplex, d_p ≥ K-1 suffices (simplex embedding)

**At our scale:** K ≈ 50-100 material classes. d_p = 32 provides ⌈log₂(100)⌉ = 7 bits for separation plus 25 dimensions for continuous variation. d_p = 64 is generous.

**Criticality:** MEDIUM — validates our choice of d_p = 32-64.

---

## 3. Lean 4 Proof Targets (for Aristotle)

Priority ordering for submission:

| Priority | Claim | Difficulty | Depends on |
|----------|-------|-----------|-----------|
| P1 (CRITICAL) | P1: Covariance SPD from quat+scale | Easy | Existing Claim 1.1 |
| P2 (CRITICAL) | P6: Correlation hypothesis (info-theoretic) | Hard | Novel |
| P3 (HIGH) | P2: Material similarity bounded [0,1] | Easy | Arithmetic |
| P4 (HIGH) | P4: Covariance similarity well-defined | Medium | P1 |
| P5 (HIGH) | P8: Densification preserves coherence | Medium | P2 |
| P6 (MEDIUM) | P5: Prediction continuity | Easy | Standard analysis |
| P7 (MEDIUM) | P9: Dimensionality sufficiency | Medium | Linear algebra |
| P8 (LOW) | P3: Material regions partition | Trivial | Graph theory |
| P9 (LOW) | P7: Collapse operators are projections | Trivial | By construction |

### Lean 4 Formalization Notes

**P1** extends existing Claim 1.1 (Cholesky PSD) from the original tracker. The key additional step: show that `R · D · R^T` with R ∈ SO(3) and D diagonal positive is SPD.

**P6** is the hardest claim and the most important. It's an information-theoretic statement. Lean 4 formalization would need: definition of mutual information, data processing inequality, conditional entropy decomposition. Mathlib has some information theory but not all. May require partial mechanization + human-guided steps.

**P2** and **P4** are straightforward arithmetic/analysis. Good candidates for clean Lean 4 proofs.

**P8** needs the Lipschitz continuity of M, which follows from the exponential/cosine terms being Lipschitz. Computable bound.

---

## 4. Notation Summary

| Symbol | Type | Meaning |
|--------|------|---------|
| G | tuple | Physical Gaussian primitive |
| p | ℝ³ | Position |
| S | ℝ³ | Log-scale |
| q | ℍ₁ | Unit quaternion (rotation) |
| α | ℝ | Opacity logit |
| c | ℝ³ | Color |
| e_s | ℝ^{d_s} | Semantic embedding |
| e_p | ℝ^{d_p} | Physical embedding |
| ℓ | string | Label |
| Σ | S³₊₊ | 3x3 SPD covariance matrix |
| R(q) | SO(3) | Rotation matrix from quaternion |
| M(·,·) | [0,1] | Material similarity |
| Φ(·,·) | (0,1] | Covariance similarity |
| f_phys | MLP | Physics prediction function |
| π_X | projection | Collapse operator for subsystem X |
| G_M | graph | Material similarity graph |
| τ | (0,1) | Clustering threshold |
| d_s | ℕ | Semantic embedding dimension (128-300) |
| d_p | ℕ | Physical embedding dimension (32-64) |

---

## 5. Connection to Existing SGS Proofs

This paper extends the following already-submitted claims:

- **Claim 1.1** (Cholesky SPD) → generalized in P1 to rotation-based parameterization
- **Claim 2.1** (Anisotropic Gaussian is Mercer kernel) → Physical Gaussian inherits this property
- **Claim 7.1** (Split halves opacity) → extended in P8 to include material coherence preservation
- **Claim 3.5** (Rendering ↔ Attention) → the collapse operator framework (P7) formalizes why the same primitive appears in different "modes"

The Physical Gaussian is a strict extension of the original SGS primitive. All prior claims remain valid. The new claims concern the added structure (e_p, M, Φ, f_phys, π_X).
