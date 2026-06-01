import Mathlib

noncomputable section

open Real Finset BigOperators

/-- The sigmoid function σ(x) = 1 / (1 + exp(-x)). -/
def sigmoid' (x : ℝ) : ℝ := 1 / (1 + exp (-x))

/-
sigmoid is strictly positive.
-/
lemma sigmoid'_pos (x : ℝ) : 0 < sigmoid' x := by
  exact one_div_pos.mpr ( add_pos zero_lt_one ( Real.exp_pos _ ) )

/-
sigmoid is strictly less than 1.
-/
lemma sigmoid'_lt_one (x : ℝ) : sigmoid' x < 1 := by
  exact div_lt_one ( by positivity ) |>.2 ( by linarith [ Real.exp_pos ( -x ) ] )

/-- sigmoid is in (0, 1). -/
lemma sigmoid'_mem_Ioo (x : ℝ) : sigmoid' x ∈ Set.Ioo (0 : ℝ) 1 :=
  ⟨sigmoid'_pos x, sigmoid'_lt_one x⟩

/-- The gated recurrence: α^{(p+1)} = α^{(p)} * sigmoid(x^{(p)}). -/
def alpha (α₀ : ℝ) (x : ℕ → ℝ) : ℕ → ℝ
  | 0 => α₀
  | p + 1 => alpha α₀ x p * sigmoid' (x p)

/-
Product formula: α^{(P)} = α^{(0)} * ∏_{p=0}^{P-1} sigmoid(x^{(p)}).
-/
theorem alpha_eq_prod (α₀ : ℝ) (x : ℕ → ℝ) (P : ℕ) :
    alpha α₀ x P = α₀ * ∏ p ∈ range P, sigmoid' (x p) := by
      induction' P with P ih;
      · aesop;
      · convert congr_arg ( fun y => y * sigmoid' ( x P ) ) ih using 1;
        rw [ Finset.prod_range_succ, mul_assoc ]

/-
If α₀ > 0, then α^{(p)} > 0 for all p.
-/
theorem alpha_pos (α₀ : ℝ) (x : ℕ → ℝ) (hα₀ : 0 < α₀) (P : ℕ) :
    0 < alpha α₀ x P := by
      exact alpha_eq_prod α₀ x P ▸ mul_pos hα₀ ( Finset.prod_pos fun _ _ => sigmoid'_pos _ )

/-
The sequence is strictly decreasing: α^{(p+1)} < α^{(p)} when α₀ > 0.
-/
theorem alpha_strict_decrease (α₀ : ℝ) (x : ℕ → ℝ) (hα₀ : 0 < α₀) (p : ℕ) :
    alpha α₀ x (p + 1) < alpha α₀ x p := by
      exact mul_lt_of_lt_one_right ( alpha_pos α₀ x hα₀ p ) ( by simpa using sigmoid'_lt_one ( x p ) )

/-
Combined: the sequence is strictly antitone.
-/
theorem alpha_strictAnti (α₀ : ℝ) (x : ℕ → ℝ) (hα₀ : 0 < α₀) :
    StrictAnti (alpha α₀ x) := by
      exact strictAnti_nat_of_succ_lt fun p => alpha_strict_decrease α₀ x hα₀ p

end