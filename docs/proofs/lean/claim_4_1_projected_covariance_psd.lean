import Mathlib

/-!
# Projected Positive Definite Matrices

We prove that if `Σ` is a `d × d` symmetric positive definite matrix and `P` is an `m × d`
matrix with rank `m` (where `m ≤ d`), then `P * Σ * Pᵀ` is `m × m` symmetric positive definite.
-/

open Matrix

/-
If `P` has rank `m` (full row rank), then `Pᵀ` has injective `mulVec`.
-/
lemma transpose_mulVec_injective_of_rank {m d : ℕ}
    (P : Matrix (Fin m) (Fin d) ℝ) (hrank : P.rank = m) :
    Function.Injective P.transpose.mulVec := by
  -- The rank of P equals m, and `Matrix.rank_transpose` gives rank(Pᵀ) = rank(P) = m.
  have htransp : (Pᵀ).rank = m := by
    rw [ Matrix.rank_transpose, hrank ];
  rw [ Matrix.rank ] at htransp;
  -- Since Pᵀ is d × m, having rank m means its columns are linearly independent. Use the connection between rank and linear independence of columns.
  have h_lin_ind : LinearMap.ker (Matrix.mulVecLin Pᵀ) = ⊥ := by
    have := LinearMap.finrank_range_add_finrank_ker ( Matrix.mulVecLin Pᵀ );
    simp_all +decide [ Submodule.eq_bot_iff ];
  exact LinearMap.ker_eq_bot.mp h_lin_ind

/-
The main theorem: projecting a positive definite matrix by a full-rank matrix
    yields a positive definite matrix.
-/
theorem posDef_projected {m d : ℕ}
    (Sigma : Matrix (Fin d) (Fin d) ℝ) (P : Matrix (Fin m) (Fin d) ℝ)
    (hS : Sigma.PosDef) (hrank : P.rank = m) :
    (P * Sigma * P.transpose).PosDef := by
  -- Since `P` has rank `m`, `P.transpose` has injective `mulVec` by `transpose_mulVec_injective_of_rank`.
  have hP_transpose_inj : Function.Injective P.transpose.mulVec := by
    exact transpose_mulVec_injective_of_rank P hrank;
  convert Matrix.PosDef.conjTranspose_mul_mul_same ( show ( Sigma |> Matrix.PosDef ) from hS ) hP_transpose_inj using 1