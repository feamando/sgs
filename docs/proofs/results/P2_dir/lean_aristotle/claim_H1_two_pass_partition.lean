import Mathlib
import Mathlib.Tactic

/-!
# Two-Pass Partition Equivalence for Alpha Compositing

Given a sequence of opacity values split into two consecutive groups ("blobs" and "words"),
rendering each group independently (with the second scaled by the residual transmittance)
produces identical results to rendering the full sequence.

We reuse the definitions of `T` (transmittance) and `w` (compositing weight) from
`claim_3_1_weights_sum_bounded.lean`, restated here for self-containedness.
-/

open Finset

noncomputable section

/-- Transmittance: `T a 0 = 1`, `T a (i+1) = T a i * (1 - a i)`. -/
def T (a : ℕ → ℝ) : ℕ → ℝ
  | 0     => 1
  | n + 1 => T a n * (1 - a n)

/-- Compositing weight: `w i = a i * T i`. -/
def w (a : ℕ → ℝ) (i : ℕ) : ℝ := a i * T a i

/-- Key identity: `w i = T i - T (i+1)`. -/
lemma w_eq_T_sub (a : ℕ → ℝ) (i : ℕ) : w a i = T a i - T a (i + 1) := by
  show a i * T a i = T a i - T a i * (1 - a i); ring

/-- Telescoping identity: `∑_{i=0}^{n-1} w(a,i) = 1 - T(a,n)`. -/
theorem alpha_compositing_sum (a : ℕ → ℝ) (n : ℕ) :
    ∑ i ∈ range n, w a i = 1 - T a n := by
  induction n with
  | zero => simp [T]
  | succ n ih => rw [sum_range_succ, ih, w_eq_T_sub]; ring

/-
Transmittance splits multiplicatively: `T(a, k+n) = T(a, k) * T(shift a k, n)`.
-/
lemma T_split (a : ℕ → ℝ) (k n : ℕ) :
    T a (k + n) = T a k * T (fun i => a (k + i)) n := by
  induction' n with n ih generalizing k;
  · norm_num [ T ];
  · grind +locals

/-
**Two-Pass Partition Equivalence**: rendering in two passes equals one pass.
-/
theorem two_pass_partition (a : ℕ → ℝ) (k n : ℕ) :
    ∑ i ∈ range (k + n), w a i =
      ∑ i ∈ range k, w a i + T a k * ∑ i ∈ range n, w (fun i => a (k + i)) i := by
  rw [ Finset.mul_sum, ← Finset.sum_range_add_sum_Ico _ ( by linarith : k ≤ k + n ) ];
  simp +decide [ w, T_split, Finset.sum_Ico_eq_sum_range ];
  grind +splitImp

end