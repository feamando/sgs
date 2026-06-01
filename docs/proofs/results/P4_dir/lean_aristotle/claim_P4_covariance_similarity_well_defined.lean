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
# Claim P4: Covariance similarity Φ is well-defined for SPD matrices

We prove that for any two symmetric positive definite matrices A, B ∈ S³₊₊:

    Φ(A, B) = exp(-½ · ||log(λ(A)) - log(λ(B))||₂) ∈ (0, 1]

and Φ(A, A) = 1, Φ(A, B) = Φ(B, A).

## Key steps

1. SPD matrices have strictly positive eigenvalues (λᵢ > 0)
2. log(λᵢ) is well-defined for positive reals
3. ||log λ(A) - log λ(B)||₂ ≥ 0 (norm is nonneg)
4. exp(-½ · t) ∈ (0, 1] for t ≥ 0
5. When A = B: ||log λ - log λ|| = 0, so Φ = exp(0) = 1
6. Symmetry: ||a - b|| = ||b - a||

## Connection to Physical Gaussians

Φ measures how "similarly shaped" two Gaussians are. Two Gaussians with
identical covariance (same shape and orientation) have Φ = 1. Gaussians
with very different shapes (one is a sphere, another is a thin disk)
have Φ → 0. This feeds into the material similarity metric M.
-/

/-- SPD matrices have positive eigenvalues. -/
theorem spd_eigenvalues_positive {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    (hA : Matrix.PosDef A) (i : Fin n) :
    0 < (Matrix.IsHermitian.eigenvalues hA.isHermitian) i := by
  convert hA.eigenvalues_pos i using 1

/-- log is well-defined on positive reals. -/
theorem log_pos_well_defined (x : ℝ) (hx : 0 < x) :
    ∃ y : ℝ, Real.exp y = x := by
  exact ⟨Real.log x, Real.exp_log hx⟩

/-- The covariance similarity is in (0, 1]. -/
theorem covariance_similarity_range (t : ℝ) (ht : 0 ≤ t) :
    0 < Real.exp (-(1/2) * t) ∧ Real.exp (-(1/2) * t) ≤ 1 := by
  constructor
  · exact Real.exp_pos _
  · have : -(1/2) * t ≤ 0 := by nlinarith
    exact Real.exp_le_one_iff.mpr this

/-
Self-similarity: Φ(A, A) = 1.
-/
theorem covariance_self_similarity (eA : Fin 3 → ℝ) (_hpos : ∀ i, 0 < eA i) :
    let log_diff_norm := ‖(fun i => Real.log (eA i)) - (fun i => Real.log (eA i))‖
    Real.exp (-(1/2) * log_diff_norm) = 1 := by
  norm_num

/-
Symmetry: Φ(A, B) = Φ(B, A).
-/
theorem covariance_similarity_symmetric (eA eB : Fin 3 → ℝ)
    (_hA : ∀ i, 0 < eA i) (_hB : ∀ i, 0 < eB i) :
    let norm_AB := ‖(fun i => Real.log (eA i) - Real.log (eB i))‖
    let norm_BA := ‖(fun i => Real.log (eB i) - Real.log (eA i))‖
    norm_AB = norm_BA := by
  convert norm_sub_rev _ _