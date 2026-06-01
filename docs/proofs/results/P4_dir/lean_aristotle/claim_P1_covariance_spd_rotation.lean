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
# Claim P1: Covariance matrix from quaternion rotation + scale is SPD

We prove that for any orthogonal matrix R ∈ SO(3) and any vector S ∈ ℝ³,
the matrix Σ = R · diag(exp(2S)) · Rᵀ is symmetric positive definite.

This extends Claim 1.1 (Cholesky PSD) to the rotation-based parameterization
used in 3D Gaussian Splatting and Physical Gaussians.

## Mathematical statement

Let R ∈ O(n) (orthogonal: R · Rᵀ = I) and D = diag(d₁, ..., dₙ) with dᵢ > 0.
Then A = R · D · Rᵀ is symmetric positive definite.

## Proof sketch

1. Symmetry: (RDRᵀ)ᵀ = R · Dᵀ · Rᵀ = R · D · Rᵀ (since D is symmetric).
2. Positive definiteness: For any x ≠ 0,
   xᵀ · R · D · Rᵀ · x = (Rᵀx)ᵀ · D · (Rᵀx) = yᵀ · D · y
   where y = Rᵀx ≠ 0 (since R is invertible).
   Since D has positive diagonal: yᵀ · D · y = Σᵢ dᵢ · yᵢ² > 0 (as y ≠ 0 and dᵢ > 0).

## Application to Physical Gaussians

In our parameterization:
- R = R(q) where q ∈ ℍ₁ is a unit quaternion (guarantees R ∈ SO(3))
- D = diag(exp(2S₁), exp(2S₂), exp(2S₃)) where Sᵢ ∈ ℝ (guarantees dᵢ = exp(2Sᵢ) > 0)

Therefore Σ = R(q) · diag(exp(2S)) · R(q)ᵀ is always SPD regardless of
the values of q and S. This means:
- The Physical Gaussian's covariance is always well-defined
- The covariance similarity Φ(Σ_i, Σ_j) is well-defined (eigenvalues exist and are positive)
- No numerical safeguards needed on the covariance itself (only on q normalization)
-/

/-- A diagonal matrix with positive entries is positive definite. -/
theorem diagonal_pos_def {n : ℕ} (d : Fin n → ℝ) (hd : ∀ i, 0 < d i) :
    Matrix.PosDef (Matrix.diagonal d) := by
  intro x hx
  simp [Matrix.dotProduct, Matrix.diagonal, Matrix.mulVec, Finset.sum_mul]
  sorry -- Aristotle: expand dotProduct, show Σᵢ dᵢ · xᵢ² > 0 given ∃ i, xᵢ ≠ 0

/-- Congruence by orthogonal matrix preserves symmetry. -/
theorem orthogonal_congruence_symmetric {n : ℕ} (R : Matrix (Fin n) (Fin n) ℝ)
    (D : Matrix (Fin n) (Fin n) ℝ)
    (hR : R * R.transpose = 1)
    (hD : D.IsSymm) :
    (R * D * R.transpose).IsSymm := by
  sorry -- Aristotle: (RDRᵀ)ᵀ = R Dᵀ Rᵀ = R D Rᵀ

/-- Congruence by invertible matrix preserves positive definiteness. -/
theorem congruence_pos_def {n : ℕ} (R : Matrix (Fin n) (Fin n) ℝ)
    (D : Matrix (Fin n) (Fin n) ℝ)
    (hR_inv : Function.Bijective R.mulVec)
    (hD : Matrix.PosDef D) :
    Matrix.PosDef (R * D * R.transpose) := by
  sorry -- Aristotle: xᵀRDRᵀx = (Rᵀx)ᵀD(Rᵀx) > 0 since Rᵀx ≠ 0

/-- Main theorem: rotation + positive diagonal scale → SPD covariance. -/
theorem physical_gaussian_covariance_spd {R : Matrix (Fin 3) (Fin 3) ℝ}
    {S : Fin 3 → ℝ}
    (hR : R * R.transpose = 1) :
    Matrix.PosDef (R * Matrix.diagonal (fun i => Real.exp (2 * S i)) * R.transpose) := by
  sorry -- Aristotle: combine diagonal_pos_def (exp > 0) with congruence_pos_def
