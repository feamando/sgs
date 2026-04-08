import Mathlib

/-!
# Alpha-Compositing Transmittance

We prove that in the alpha-compositing sequence
  T₁ = 1, T_{i+1} = T_i · (1 - a_i)
where each a_i ∈ [0, 1], the transmittance is non-increasing and non-negative:
  T₁ ≥ T₂ ≥ ⋯ ≥ T_{n+1} ≥ 0.
-/

noncomputable section

/-- The transmittance sequence: T 0 = 1, T (i+1) = T i * (1 - a i). -/
def transmittance (a : ℕ → ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => transmittance a n * (1 - a n)

/-
Each transmittance value is non-negative when all alphas are in [0, 1].
-/
theorem transmittance_nonneg (a : ℕ → ℝ) (ha : ∀ i, a i ∈ Set.Icc 0 1) :
    ∀ n, 0 ≤ transmittance a n := by
      intro n;
      induction' n with n ih;
      · exact zero_le_one;
      · exact mul_nonneg ih ( sub_nonneg.2 <| ha n |>.2 )

/-
The transmittance sequence is non-increasing when all alphas are in [0, 1].
-/
theorem transmittance_mono (a : ℕ → ℝ) (ha : ∀ i, a i ∈ Set.Icc 0 1) :
    ∀ n, transmittance a (n + 1) ≤ transmittance a n := by
      exact fun n => mul_le_of_le_one_right ( transmittance_nonneg a ha n ) ( sub_le_self _ ( ha n |>.1 ) )

end