import Mathlib

open scoped BigOperators
open EuclideanSpace

set_option maxHeartbeats 8000000

/-!
# Linear Growth Bound for Iterative Tanh Updates

We prove that for an iterative update of the form:
  μ^{(p+1)} = μ^{(p)} + tanh(W · h^{(p)} + b)
where tanh outputs values in (-1,1)^d, the norm grows at most linearly:
  ‖μ^{(P)}‖ ≤ ‖μ^{(0)}‖ + P · √d
-/

/-
The norm of a vector in ℝ^d with all components having absolute value < 1
    is at most √d. This captures the key property of the tanh activation function.
-/
lemma norm_le_sqrt_of_components_lt_one (d : ℕ) (v : EuclideanSpace ℝ (Fin d))
    (hv : ∀ i, |v i| < 1) : ‖v‖ ≤ Real.sqrt d := by
      rw [ EuclideanSpace.norm_eq ];
      exact Real.sqrt_le_sqrt <| le_trans ( Finset.sum_le_sum fun _ _ => pow_le_one₀ ( abs_nonneg _ ) ( le_of_lt ( hv _ ) ) ) ( by norm_num )

/-
**Linear growth bound for iterative tanh updates.**

Given a sequence μ^{(p)} in ℝ^d updated by:
  μ^{(p+1)} = μ^{(p)} + δ^{(p)}
where each δ^{(p)} has all components bounded in absolute value by 1
(as is the case for tanh outputs), the norm satisfies:
  ‖μ^{(P)}‖ ≤ ‖μ^{(0)}‖ + P · √d

That is, the position grows at most linearly with the number of iterations.
-/
theorem iterative_tanh_linear_growth (d : ℕ) (P : ℕ)
    (mu : ℕ → EuclideanSpace ℝ (Fin d))
    (delta : ℕ → EuclideanSpace ℝ (Fin d))
    (h_update : ∀ p, mu (p + 1) = mu p + delta p)
    (h_bound : ∀ p i, |delta p i| < 1) :
    ‖mu P‖ ≤ ‖mu 0‖ + P * Real.sqrt d := by
      induction' P with P ih;
      · norm_num;
      · rw [ h_update ];
        refine' le_trans ( norm_add_le _ _ ) _;
        convert add_le_add ih ( norm_le_sqrt_of_components_lt_one d ( delta P ) fun i => h_bound P i ) using 1 ; push_cast ; ring