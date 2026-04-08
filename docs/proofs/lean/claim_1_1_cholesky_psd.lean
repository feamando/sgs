import Mathlib

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

set_option pp.fullNames true
set_option pp.structureInstances true
set_option pp.coercions.types true
set_option pp.funBinderTypes true
set_option pp.letVarTypes true
set_option pp.piBinderTypes true

set_option grind.warning false

/-!
# Lower-triangular matrices with positive diagonal: LLᵀ + εI is symmetric positive definite

We prove that for any lower-triangular matrix `L` over `ℝ` with positive diagonal entries,
and any `ε > 0`, the matrix `L * Lᵀ + ε • I` is symmetric positive definite.

The key insight is that `L * Lᵀ` is always positive semidefinite (for any matrix `L`),
and `ε • I` is positive definite when `ε > 0`. The sum of a positive semidefinite and a
positive definite matrix is positive definite.

The lower-triangular and positive-diagonal hypotheses are included as stated in the problem,
though the result holds for any matrix `L`.
-/

/-- A matrix is lower-triangular if all entries above the diagonal are zero. -/
def Matrix.IsLowerTriangular {d : Type*} [LT d] (L : Matrix d d ℝ) : Prop :=
  ∀ i j, i < j → L i j = 0

/-- A matrix has positive diagonal entries. -/
def Matrix.HasPosDiag {d : Type*} (L : Matrix d d ℝ) : Prop :=
  ∀ i, 0 < L i i

/-- For any lower-triangular matrix `L` with positive diagonal entries and any `ε > 0`,
the matrix `L * Lᵀ + ε • I` is symmetric positive definite. -/
theorem lower_triangular_LLT_add_eps_posDef
    {d : Type*} [Fintype d] [DecidableEq d] [LinearOrder d]
    (L : Matrix d d ℝ) (ε : ℝ)
    (_hL_lower : Matrix.IsLowerTriangular L)
    (_hL_diag : Matrix.HasPosDiag L)
    (hε : 0 < ε) :
    (L * L.transpose + ε • (1 : Matrix d d ℝ)).PosDef := by
  apply Matrix.PosDef.posSemidef_add
  · convert Matrix.posSemidef_self_mul_conjTranspose L using 1
  · exact Matrix.PosDef.one.smul hε
