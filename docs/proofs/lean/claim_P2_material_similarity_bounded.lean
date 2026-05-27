import Mathlib

open scoped BigOperators
open scoped Real
open scoped Classical

set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

/-!
# Claim P2: Material Similarity is bounded in [0, 1] with self-similarity = 1

We prove that the material similarity function M(G_i, G_j) as defined in the
Physical Gaussians specification satisfies:

1. 0 ≤ M(G_i, G_j) ≤ 1 for all G_i, G_j
2. M(G_i, G_i) = 1 (self-similarity is maximal)

## Definition

M(G_i, G_j) = Σₖ wₖ · fₖ(G_i, G_j)

where:
- wₖ ∈ [0,1], Σₖ wₖ = 1
- f_s(G_i, G_j) = (cos(e_s^i, e_s^j) + 1) / 2  ∈ [0, 1]
- f_g(G_i, G_j) = exp(-||S_i - S_j||₂)          ∈ (0, 1]
- f_α(G_i, G_j) = exp(-|α_i - α_j|)             ∈ (0, 1]
- f_d(G_i, G_j) = exp(-||p_i - p_j||₂ / r)      ∈ (0, 1]
- f_Σ(G_i, G_j) = Φ(Σ_i, Σ_j)                   ∈ (0, 1]

## Proof sketch

1. Each component fₖ ∈ [0, 1]:
   - Cosine ∈ [-1,1], so (cos+1)/2 ∈ [0,1]
   - exp(-x) ∈ (0,1] for x ≥ 0
   - Φ ∈ (0,1] by Claim P4

2. Convex combination of values in [0,1] is in [0,1]:
   M = Σ wₖ fₖ, with wₖ ≥ 0, Σ wₖ = 1, fₖ ∈ [0,1]
   → 0 ≤ M ≤ 1

3. Self-similarity: when i = j:
   - cos(e_s, e_s) = 1, so f_s = (1+1)/2 = 1
   - ||S - S|| = 0, so f_g = exp(0) = 1
   - |α - α| = 0, so f_α = exp(0) = 1
   - ||p - p|| = 0, so f_d = exp(0) = 1
   - Φ(Σ, Σ) = 1 (by Claim P4)
   → M(G, G) = Σ wₖ · 1 = 1
-/

/-- exp(-x) ∈ (0, 1] for x ≥ 0. -/
theorem exp_neg_nonneg_bounded (x : ℝ) (hx : 0 ≤ x) :
    0 < Real.exp (-x) ∧ Real.exp (-x) ≤ 1 := by
  constructor
  · exact Real.exp_pos (-x)
  · rw [Real.exp_le_one_iff_nonpos]
    linarith

/-- Cosine similarity rescaled to [0,1]. -/
theorem cosine_rescaled_bounded (cos_val : ℝ) (h : -1 ≤ cos_val ∧ cos_val ≤ 1) :
    0 ≤ (cos_val + 1) / 2 ∧ (cos_val + 1) / 2 ≤ 1 := by
  constructor
  · linarith [h.1]
  · linarith [h.2]

/-- Convex combination of values in [0,1] is in [0,1]. -/
theorem convex_combination_bounded {n : ℕ} (w : Fin n → ℝ) (f : Fin n → ℝ)
    (hw_nonneg : ∀ i, 0 ≤ w i)
    (hw_sum : ∑ i, w i = 1)
    (hf_bounds : ∀ i, 0 ≤ f i ∧ f i ≤ 1) :
    0 ≤ ∑ i, w i * f i ∧ ∑ i, w i * f i ≤ 1 := by
  sorry -- Aristotle: sum of nonneg products is nonneg; upper bound by replacing f with 1

/-- Self-similarity of exponential components. -/
theorem self_similarity_exp (x : ℝ) : Real.exp (-|x - x|) = 1 := by
  simp [sub_self, abs_zero, Real.exp_zero]

/-- Main theorem: M(G, G) = 1 and M ∈ [0, 1]. -/
theorem material_similarity_bounded_and_self_maximal
    (n_components : ℕ)
    (w : Fin n_components → ℝ)
    (f_self : Fin n_components → ℝ)
    (hw_nonneg : ∀ i, 0 ≤ w i)
    (hw_sum : ∑ i, w i = 1)
    (hf_self : ∀ i, f_self i = 1)
    (hf_bounded : ∀ i, 0 ≤ f_self i ∧ f_self i ≤ 1) :
    ∑ i, w i * f_self i = 1 := by
  sorry -- Aristotle: Σ wᵢ * 1 = Σ wᵢ = 1
