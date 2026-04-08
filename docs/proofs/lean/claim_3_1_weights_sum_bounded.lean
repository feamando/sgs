import Mathlib

/-!
# Alpha Compositing / Volume Rendering Identity

Given scalars `a_1, ..., a_n ∈ [0,1]`, define:
- `T 1 = 1` and `T (i+1) = T i * (1 - a i)` (transmittance)
- `w i = a i * T i` (compositing weight)

We prove:
1. `∑_{i=1}^{n} w_i = 1 - T_{n+1}` (telescoping identity)
2. `0 ≤ ∑ w_i ≤ 1` (weights are bounded)
3. `∑ w_i = 1 ↔ T_{n+1} = 0` (full opacity characterization)
-/

open Finset

noncomputable section

/-- Transmittance: `T 1 = 1`, `T (i+1) = T i * (1 - a i)`. We use 0-indexed internally:
    `T a 0 = 1`, `T a (i+1) = T a i * (1 - a i)`. -/
def T (a : ℕ → ℝ) : ℕ → ℝ
  | 0     => 1
  | n + 1 => T a n * (1 - a n)

/-- Compositing weight: `w i = a i * T i`. -/
def w (a : ℕ → ℝ) (i : ℕ) : ℝ := a i * T a i

/-
Key algebraic identity: `w i = T i - T (i+1)`.
-/
lemma w_eq_T_sub (a : ℕ → ℝ) (i : ℕ) : w a i = T a i - T a (i + 1) := by
  exact show a i * T a i = T a i - T a i * ( 1 - a i ) by ring

/-
T is nonneg when all alphas are in [0,1].
-/
lemma T_nonneg (a : ℕ → ℝ) (ha : ∀ i, a i ∈ Set.Icc (0 : ℝ) 1) (n : ℕ) :
    0 ≤ T a n := by
  induction' n with n ih;
  · exact zero_le_one;
  · exact mul_nonneg ih ( sub_nonneg.2 <| ha n |>.2 )

/-
T is at most 1 when all alphas are in [0,1].
-/
lemma T_le_one (a : ℕ → ℝ) (ha : ∀ i, a i ∈ Set.Icc (0 : ℝ) 1) (n : ℕ) :
    T a n ≤ 1 := by
  induction' n with n ih;
  · exact le_rfl;
  · exact mul_le_one₀ ih ( sub_nonneg.2 <| ha n |>.2 ) ( sub_le_self _ <| ha n |>.1 )

/-
**(1) Telescoping identity**: `∑_{i=0}^{n-1} w_i = 1 - T n`.
-/
theorem alpha_compositing_sum (a : ℕ → ℝ) (n : ℕ) :
    ∑ i ∈ range n, w a i = 1 - T a n := by
  induction n <;> simp_all +decide [ Finset.sum_range_succ, w_eq_T_sub ]
  · exact Eq.symm (sub_self 1)
  · linarith

/-
**(2a)** The sum of weights is nonnegative.
-/
theorem alpha_compositing_sum_nonneg (a : ℕ → ℝ) (ha : ∀ i, a i ∈ Set.Icc (0 : ℝ) 1)
    (n : ℕ) : 0 ≤ ∑ i ∈ range n, w a i := by
  rw [ alpha_compositing_sum ];
  exact sub_nonneg_of_le ( T_le_one a ha n )

/-
**(2b)** The sum of weights is at most 1.
-/
theorem alpha_compositing_sum_le_one (a : ℕ → ℝ) (ha : ∀ i, a i ∈ Set.Icc (0 : ℝ) 1)
    (n : ℕ) : ∑ i ∈ range n, w a i ≤ 1 := by
  rw [ alpha_compositing_sum a n ];
  exact sub_le_self _ ( T_nonneg a ha n )

/-
**(3)** The sum of weights equals 1 iff the final transmittance is 0.
-/
theorem alpha_compositing_sum_eq_one_iff (a : ℕ → ℝ) (_ha : ∀ i, a i ∈ Set.Icc (0 : ℝ) 1)
    (n : ℕ) : ∑ i ∈ range n, w a i = 1 ↔ T a n = 0 := by
  rw [ alpha_compositing_sum ] ; aesop

end