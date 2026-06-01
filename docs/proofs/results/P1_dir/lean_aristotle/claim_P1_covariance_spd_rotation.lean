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
  exact Matrix.PosDef.diagonal hd

/-
Congruence by orthogonal matrix preserves symmetry.
-/
theorem orthogonal_congruence_symmetric {n : ℕ} (R : Matrix (Fin n) (Fin n) ℝ)
    (D : Matrix (Fin n) (Fin n) ℝ)
    (hR : R * R.transpose = 1)
    (hD : D.IsSymm) :
    (R * D * R.transpose).IsSymm := by
  simp_all +decide [ Matrix.IsSymm, Matrix.mul_assoc ]

/-
Congruence by invertible matrix preserves positive definiteness.
-/
theorem congruence_pos_def {n : ℕ} (R : Matrix (Fin n) (Fin n) ℝ)
    (D : Matrix (Fin n) (Fin n) ℝ)
    (hR_inv : Function.Bijective R.mulVec)
    (hD : Matrix.PosDef D) :
    Matrix.PosDef (R * D * R.transpose) := by
  constructor <;> simp_all +decide [ Matrix.IsHermitian ];
  · simp +decide only [Matrix.PosDef, Matrix.IsHermitian, Matrix.mul_assoc] at hD ⊢
    exact Matrix.IsSymm.eq hD.1 ▸ rfl;
  · intro x hx_ne
    have h_pos : 0 < (R.transpose.mulVec (x.toFun)) ⬝ᵥ (D.mulVec (R.transpose.mulVec (x.toFun))) := by
      have h_pos : ∀ y : Fin n → ℝ, y ≠ 0 → 0 < y ⬝ᵥ (D.mulVec y) := by
        intro y hy_ne; have := hD.2; simp_all +decide [ Matrix.mulVec, dotProduct ] ;
        convert this ( show ( Finsupp.equivFunOnFinite.symm y ) ≠ 0 from by simpa [ funext_iff, Finsupp.ext_iff ] using hy_ne ) using 1 ; simp +decide [ Finsupp.sum_fintype, Finset.mul_sum _ _ _, mul_assoc, mul_comm, mul_left_comm ];
      apply h_pos; intro h_eq_zero; (
      have := hR_inv.1 ( show R.mulVec ( fun i => x i ) = R.mulVec 0 from ?_ ) ; simp_all +decide [ funext_iff ] ;
      · exact hx_ne <| Finsupp.ext this;
      · have h_eq_zero : ∀ y : Fin n → ℝ, (R.transpose.mulVec y) = 0 → y = 0 := by
          intro y hy; have := hR_inv.1; simp_all +decide [ funext_iff, Matrix.mulVec ] ;
          have := Matrix.eq_zero_of_mulVec_eq_zero ( show Matrix.det ( R.transpose ) ≠ 0 from ?_ ) ( show Matrix.mulVec ( R.transpose ) y = 0 from ?_ ) ; simp_all +decide [ funext_iff, Matrix.mulVec ] ;
          · intro h; have := Matrix.exists_mulVec_eq_zero_iff.mpr h; obtain ⟨ v, hv ⟩ := this; have := @this; simp_all +decide [ funext_iff, Matrix.mulVec ] ;
            exact absurd ( Matrix.exists_mulVec_eq_zero_iff.mpr h ) ( by intro H; obtain ⟨ w, hw ⟩ := H; have := @this w 0; aesop );
          · exact funext fun i => by simpa [ Matrix.mulVec, dotProduct ] using hy i;
        exact congr_arg _ ( h_eq_zero _ ‹_› ));
    simp_all +decide [ Matrix.mul_assoc, Matrix.dotProduct_mulVec, Matrix.vecMul_mulVec ];
    convert h_pos using 1 ; simp +decide [ Matrix.vecMul, dotProduct, Finset.mul_sum _ _ _, mul_assoc, mul_comm, mul_left_comm, Finsupp.sum_fintype ];
    exact Finset.sum_comm.trans ( Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by ring! )

/-
Main theorem: rotation + positive diagonal scale → SPD covariance.
-/
theorem physical_gaussian_covariance_spd {R : Matrix (Fin 3) (Fin 3) ℝ}
    {S : Fin 3 → ℝ}
    (hR : R * R.transpose = 1) :
    Matrix.PosDef (R * Matrix.diagonal (fun i => Real.exp (2 * S i)) * R.transpose) := by
  convert congruence_pos_def R ( Matrix.diagonal fun i => Real.exp ( 2 * S i ) ) _ _ using 1;
  · have h_inv : Invertible R := by
      exact invertibleOfRightInverse _ _ hR;
    exact ⟨ fun x y hxy => by simpa using congr_arg ( fun z => R⁻¹.mulVec z ) hxy, fun x => ⟨ R⁻¹.mulVec x, by simp +decide ⟩ ⟩;
  · exact diagonal_pos_def _ fun i => Real.exp_pos _