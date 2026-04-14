import Mathlib
import claim_3_5_softmax_subset_alpha

open Finset BigOperators Real

noncomputable section

/-!
# Expressiveness Preservation Under Transmittance Scaling (Claim H2)

We prove that scaling all alpha compositing weights by a constant T₀ ∈ (0, 1]
preserves relative weights. The construction prepends a "budget-consuming" element
with opacity (1 - T₀), which absorbs exactly (1 - T₀) of the transmittance budget,
leaving T₀ for the remaining elements.

## Main result

`transmittance_scaling`: Given any weight vector w with Σ w_i = 1 and all w_i > 0,
  there exist alpha compositing parameters (of length n+2) such that the first element
  consumes (1 - T₀) of the budget and each subsequent element produces T₀ * w_i.
-/

variable {n : ℕ}

/-- Scaled alpha values: prepend a budget-consuming element with opacity (1 - T₀),
  then use the same constructAlpha values as in the unscaled case. -/
def scaledAlpha (T₀ : ℝ) (w : Fin (n + 1) → ℝ) : Fin (n + 2) → ℝ :=
  fun i =>
    if h : i.val = 0 then 1 - T₀
    else constructAlpha w ⟨i.val - 1, by omega⟩

/-
---------------------------------------------------------------------
Section 1: The budget-consuming element gets weight (1 - T₀)
---------------------------------------------------------------------
-/

lemma scaledAlpha_zero (T₀ : ℝ) (w : Fin (n + 1) → ℝ) :
    scaledAlpha T₀ w ⟨0, by omega⟩ = 1 - T₀ := by
  simp [scaledAlpha]

lemma scaledAlpha_succ (T₀ : ℝ) (w : Fin (n + 1) → ℝ) (i : Fin (n + 1)) :
    scaledAlpha T₀ w ⟨i.val + 1, by omega⟩ = constructAlpha w i := by
  simp [scaledAlpha]

lemma alphaWeight_scaledAlpha_zero (T₀ : ℝ) (w : Fin (n + 1) → ℝ) :
    alphaWeight (scaledAlpha T₀ w) ⟨0, by omega⟩ = 1 - T₀ := by
  exact mul_one _

/-
---------------------------------------------------------------------
Section 2: Subsequent elements get weight T₀ * w_i
---------------------------------------------------------------------

Key insight: The product ∏_{j < i+1} (1 - scaledAlpha(j)) factors as
  (1 - scaledAlpha(0)) * ∏_{j=1}^{i} (1 - scaledAlpha(j))
  = T₀ * ∏_{j < i} (1 - constructAlpha(w, j))
  = T₀ * tailSum(w, i)

So alphaWeight(scaledAlpha, i+1) = constructAlpha(w, i) * T₀ * tailSum(w, i)
  = (w_i / tailSum(w, i)) * T₀ * tailSum(w, i) = T₀ * w_i.

The transmittance product for scaledAlpha at position i+1 equals
    T₀ times the transmittance product for constructAlpha at position i.
-/
lemma prod_scaledAlpha_succ (T₀ : ℝ) (w : Fin (n + 1) → ℝ) (i : Fin (n + 1)) :
    ∏ j ∈ Finset.Iio ⟨i.val + 1, by omega⟩,
      (1 - scaledAlpha T₀ w j) =
    T₀ * ∏ j ∈ Finset.Iio i, (1 - constructAlpha w j) := by
  induction' i with i ih;
  induction' i with i ih;
  · erw [ Finset.prod_eq_single 0 ] <;> norm_num;
    erw [ Finset.prod_empty, mul_one, scaledAlpha ] ; norm_num;
  · rw [ show ( Iio ⟨ i + 1 + 1, by linarith ⟩ : Finset ( Fin ( n + 2 ) ) ) = Insert.insert ⟨ i + 1, by linarith ⟩ ( Iio ⟨ i + 1, by linarith ⟩ ) from ?_, Finset.prod_insert ] <;> norm_num;
    · rw [ show ( Iio ⟨ i + 1, ih ⟩ : Finset ( Fin ( n + 1 ) ) ) = Insert.insert ⟨ i, by linarith ⟩ ( Iio ⟨ i, by linarith ⟩ ) from ?_, Finset.prod_insert ] <;> norm_num;
      · grind +locals;
      · ext ⟨ j, hj ⟩ ; simp +decide [ Nat.lt_succ_iff ];
    · ext ⟨ j, hj ⟩ ; simp +decide [ Nat.lt_succ_iff ]

/-
The main scaling identity: each subsequent element produces T₀ * w_i.
-/
lemma alphaWeight_scaledAlpha_succ (T₀ : ℝ) (w : Fin (n + 1) → ℝ)
    (hw : ∀ i, 0 < w i) (hsum : ∑ i, w i = 1)
    (i : Fin (n + 1)) :
    alphaWeight (scaledAlpha T₀ w) ⟨i.val + 1, by omega⟩ = T₀ * w i := by
  -- By definition of `alphaWeight`, we have:
  have h_alpha_weight : alphaWeight (scaledAlpha T₀ w) ⟨i.val + 1, by omega⟩ = scaledAlpha T₀ w ⟨i.val + 1, by omega⟩ * ∏ j ∈ Finset.Iio ⟨i.val + 1, by omega⟩, (1 - scaledAlpha T₀ w j) := by
    rfl;
  rw [ h_alpha_weight, prod_scaledAlpha_succ, scaledAlpha_succ ];
  convert congr_arg ( fun x : ℝ => T₀ * x ) ( alphaWeight_constructAlpha w hw hsum i ) using 1;
  unfold alphaWeight; ring;

/-
---------------------------------------------------------------------
Section 3: Relative proportions are preserved
---------------------------------------------------------------------
-/
lemma relative_weights_preserved (T₀ : ℝ) (hT₀ : 0 < T₀)
    (w_i w_j : ℝ) (hw_j : w_j ≠ 0) :
    (T₀ * w_i) / (T₀ * w_j) = w_i / w_j := by
  rw [ mul_div_mul_left _ _ hT₀.ne' ]

/-
---------------------------------------------------------------------
Section 4: Main theorem — transmittance scaling preserves expressiveness
---------------------------------------------------------------------

**Claim H2**: Transmittance scaling preserves expressiveness.
  For any T₀ ∈ (0, 1] and any weight vector w with Σ w_i = 1 and all w_i > 0,
  there exist alpha compositing parameters of length n+2 such that:
  - The first element consumes (1 - T₀) of the transmittance budget
  - Each subsequent element i+1 produces weight T₀ * w_i
  - All alpha values lie in [0, 1]
  - The relative proportions w_i / w_j are preserved
-/
theorem transmittance_scaling
    (T₀ : ℝ) (hT₀_pos : 0 < T₀) (hT₀_le : T₀ ≤ 1)
    (w : Fin (n + 1) → ℝ) (hw : ∀ i, 0 < w i) (hsum : ∑ i, w i = 1) :
    ∃ (a : Fin (n + 2) → ℝ),
      (∀ i, a i ∈ Set.Icc (0 : ℝ) 1) ∧
      alphaWeight a ⟨0, by omega⟩ = 1 - T₀ ∧
      (∀ i : Fin (n + 1), alphaWeight a ⟨i.val + 1, by omega⟩ = T₀ * w i) ∧
      (∀ i j : Fin (n + 1), w j ≠ 0 →
        (T₀ * w i) / (T₀ * w j) = w i / w j) := by
  refine' ⟨ scaledAlpha T₀ w, _, _, _, _ ⟩;
  · intro i;
    by_cases hi : i.val = 0;
    · simp_all +decide [ scaledAlpha ];
      linarith;
    · convert constructAlpha_mem_Icc w hw hsum ⟨ i.val - 1, _ ⟩;
      exact if_neg hi;
  · grind +suggestions;
  · exact fun i => alphaWeight_scaledAlpha_succ T₀ w hw hsum i;
  · exact fun i j hj => mul_div_mul_left _ _ hT₀_pos.ne'

end