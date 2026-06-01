import Mathlib

open Finset BigOperators Real

noncomputable section

/-!
# Softmax Attention vs Alpha Compositing

We investigate the relationship between two weighted aggregation schemes:
- **Scheme A (Softmax)**: weights `w_i = exp(s_i) / Σ_j exp(s_j)`
- **Scheme B (Alpha Compositing)**: weights `w_i = a_i * Π_{j<i} (1 - a_j)`

## Main results

1. `alpha_not_subset_softmax`: Alpha compositing is NOT a special case of softmax,
   because alpha compositing can produce zero weights while softmax weights are always
   strictly positive.
2. `softmax_subset_alpha`: Softmax IS a special case of alpha compositing.
   Given any softmax weight vector, we can construct alpha values that exactly reproduce it.
   The construction is `a_i = w_i / Σ_{j≥i} w_j`, which telescopes to recover the
   original weights.
-/

variable {n : ℕ}

/-- Softmax attention weight for index `i` given scores `s`. -/
def softmaxWeight (s : Fin n → ℝ) (i : Fin n) : ℝ :=
  exp (s i) / ∑ j : Fin n, exp (s j)

/-- Alpha compositing weight for index `i` given opacity values `a`.
  Each layer `i` contributes `a_i` attenuated by the transmittance `∏_{j<i} (1 - a_j)`. -/
def alphaWeight (a : Fin n → ℝ) (i : Fin n) : ℝ :=
  a i * ∏ j ∈ Finset.Iio i, (1 - a j)

/-- Tail sum: `Σ_{j ≥ i} w_j`. -/
def tailSum (w : Fin n → ℝ) (i : Fin n) : ℝ :=
  ∑ j ∈ Finset.Ici i, w j

/-- Construct alpha compositing parameters from a weight vector.
  Given weights `w`, set `a_i = w_i / tailSum(w, i)`. -/
def constructAlpha (w : Fin n → ℝ) (i : Fin n) : ℝ :=
  w i / tailSum w i

/-
---------------------------------------------------------------------
Section 1: Basic properties of softmax
---------------------------------------------------------------------
-/
lemma sum_exp_pos (s : Fin (n + 1) → ℝ) : 0 < ∑ j : Fin (n + 1), exp (s j) := by
  exact Finset.sum_pos ( fun _ _ => Real.exp_pos _ ) ( Finset.univ_nonempty )

lemma softmaxWeight_pos (s : Fin (n + 1) → ℝ) (i : Fin (n + 1)) :
    0 < softmaxWeight s i := by
  exact div_pos ( Real.exp_pos _ ) ( sum_exp_pos s )

lemma softmaxWeight_sum (s : Fin (n + 1) → ℝ) :
    ∑ i : Fin (n + 1), softmaxWeight s i = 1 := by
  unfold softmaxWeight;
  rw [ ← Finset.sum_div _ _ _, div_self <| ne_of_gt <| sum_exp_pos s ]

/-
---------------------------------------------------------------------
Section 2: Alpha compositing is NOT a special case of softmax
---------------------------------------------------------------------

Alpha compositing can produce weight vectors that no softmax can produce.
  Specifically, taking `a = ![0, 1]` gives the weight vector `(0, 1)`,
  but softmax weights are always strictly positive.
-/
theorem alpha_not_subset_softmax :
    ∃ (a : Fin 2 → ℝ), (∀ i, a i ∈ Set.Icc (0 : ℝ) 1) ∧
      ¬∃ (s : Fin 2 → ℝ), ∀ i, alphaWeight a i = softmaxWeight s i := by
  -- Consider the case where `a = ![0, 1]`.
  use ![0, 1];
  simp_all +decide [ Fin.forall_fin_two, alphaWeight ];
  exact fun x hx => absurd hx <| ne_of_lt <| softmaxWeight_pos x 0

/-
---------------------------------------------------------------------
Section 3: Softmax IS a special case of alpha compositing
---------------------------------------------------------------------
-/
lemma tailSum_pos (w : Fin (n + 1) → ℝ) (hw : ∀ i, 0 < w i) (i : Fin (n + 1)) :
    0 < tailSum w i := by
  exact Finset.sum_pos ( fun j hj => hw j ) ( by simp )

lemma tailSum_zero_eq (w : Fin (n + 1) → ℝ) :
    tailSum w ⟨0, Nat.zero_lt_succ n⟩ = ∑ j : Fin (n + 1), w j := by
  exact Finset.sum_subset ( Finset.subset_univ _ ) fun x hx₁ hx₂ => by aesop

lemma tailSum_succ (w : Fin (n + 1) → ℝ) (i : Fin (n + 1)) (hi : i.val + 1 ≤ n) :
    tailSum w i = w i + tailSum w ⟨i.val + 1, by omega⟩ := by
  unfold tailSum;
  rw [ Finset.Ici_eq_cons_Ioi, Finset.sum_cons ];
  rcongr j ; aesop

lemma constructAlpha_mem_Icc (w : Fin (n + 1) → ℝ) (hw : ∀ i, 0 < w i)
    (_hsum : ∑ i : Fin (n + 1), w i = 1) (i : Fin (n + 1)) :
    constructAlpha w i ∈ Set.Icc (0 : ℝ) 1 := by
  refine' ⟨ div_nonneg ( le_of_lt ( hw i ) ) _, _ ⟩;
  · exact Finset.sum_nonneg fun _ _ => le_of_lt ( hw _ );
  · refine' div_le_one_of_le₀ _ ( le_of_lt ( tailSum_pos _ hw i ) );
    exact Finset.single_le_sum ( fun a _ => le_of_lt ( hw a ) ) ( by simp )

/-
The key telescoping identity: the transmittance through layers `0, ..., i-1`
  equals the tail sum from `i` onward.
  Proof by induction on `i.val`:
  - Base `i = 0`: empty product = 1 = tailSum at 0 = total sum = 1.
  - Step: product up to `i+1` = (product up to `i`) * (1 - a_i)
    = tailSum(i) * (1 - w_i/tailSum(i)) = tailSum(i) - w_i = tailSum(i+1).
-/
lemma prod_one_sub_constructAlpha (w : Fin (n + 1) → ℝ) (hw : ∀ i, 0 < w i)
    (hsum : ∑ i : Fin (n + 1), w i = 1) (i : Fin (n + 1)) :
    ∏ j ∈ Finset.Iio i, (1 - constructAlpha w j) = tailSum w i := by
  induction' i using Fin.induction with i ih;
  · unfold constructAlpha tailSum;
    erw [ Finset.prod_empty, show ( Ici 0 : Finset ( Fin ( n + 1 ) ) ) = Finset.univ from Finset.eq_univ_of_forall fun i => Finset.mem_Ici.mpr ( Nat.zero_le _ ) ] ; aesop;
  · rw [ show ( Finset.Iio ( Fin.succ i ) : Finset ( Fin ( n + 1 ) ) ) = Finset.Iio ( Fin.castSucc i ) ∪ { Fin.castSucc i } from ?_, Finset.prod_union ] <;> norm_num;
    · rw [ ih, tailSum_succ ];
      rw [ constructAlpha ];
      all_goals norm_num;
      rw [ one_sub_div, mul_div, mul_comm ];
      · rw [ div_eq_iff ];
        · rw [ tailSum_succ ] ; ring!;
          exact Nat.succ_le_of_lt i.2;
        · exact ne_of_gt <| Finset.sum_pos ( fun _ _ => hw _ ) <| Finset.nonempty_Ici;
      · exact ne_of_gt <| tailSum_pos _ ( fun i => hw i ) _;
    · exact val_inj.mp rfl

lemma alphaWeight_constructAlpha (w : Fin (n + 1) → ℝ) (hw : ∀ i, 0 < w i)
    (hsum : ∑ i : Fin (n + 1), w i = 1) (i : Fin (n + 1)) :
    alphaWeight (constructAlpha w) i = w i := by
  convert congr_arg ( fun x : ℝ => w i / tailSum w i * x ) ( prod_one_sub_constructAlpha w hw hsum i ) using 1;
  rw [ div_mul_cancel₀ _ ( ne_of_gt ( tailSum_pos w hw i ) ) ]

/-
Softmax is a special case of alpha compositing: every softmax weight vector can be
  exactly represented as alpha compositing weights.
-/
theorem softmax_subset_alpha (s : Fin (n + 1) → ℝ) :
    ∃ (a : Fin (n + 1) → ℝ), (∀ i, a i ∈ Set.Icc (0 : ℝ) 1) ∧
      ∀ i, alphaWeight a i = softmaxWeight s i := by
  apply Exists.intro (fun i => constructAlpha (softmaxWeight s) i);
  apply And.intro;
  · exact fun i => constructAlpha_mem_Icc _ ( fun i => softmaxWeight_pos _ _ ) ( by simp [softmaxWeight_sum] ) _;
  · apply alphaWeight_constructAlpha;
    · exact fun i => softmaxWeight_pos s i;
    · exact softmaxWeight_sum s

end