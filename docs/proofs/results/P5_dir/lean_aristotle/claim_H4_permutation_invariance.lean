import Mathlib

/-!
# Claim H4: Total Rendering Weight is Permutation-Invariant

Given scalars `a_0, ..., a_{n-1} ∈ [0,1]`, define:
- `T a 0 = 1` and `T a (i+1) = T a i * (1 - a i)` (transmittance)
- `w a i = a i * T a i` (compositing weight)

We prove:
1. The transmittance `T a n` equals the finite product `∏ i in range n, (1 - a i)`.
2. The total sum `∑ w(a, i)` is invariant under any permutation of the opacity values.
3. (Corollary) For two elements, the total weight `a₁ + a₂ - a₁ * a₂` is symmetric.

The key insight is that blob ordering affects how weight is distributed across individual
elements, but not the total rendering budget consumed.
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
  exact show a i * T a i = T a i - T a i * (1 - a i) by ring

/-- Telescoping identity: `∑_{i=0}^{n-1} w_i = 1 - T n`. -/
theorem alpha_compositing_sum (a : ℕ → ℝ) (n : ℕ) :
    ∑ i ∈ range n, w a i = 1 - T a n := by
  induction n <;> simp_all +decide [Finset.sum_range_succ, w_eq_T_sub]
  · exact Eq.symm (sub_self 1)
  · linarith

/-! ## Transmittance as a finite product -/

/-- Transmittance equals the finite product `∏ i in range n, (1 - a i)`. -/
lemma T_eq_prod (a : ℕ → ℝ) (n : ℕ) :
    T a n = ∏ i ∈ range n, (1 - a i) := by
  induction n with
  | zero => simp [T]
  | succ n ih => simp [T, Finset.prod_range_succ, ih]

/-! ## Permutation invariance of total weight -/

/-- Permute the first `n` opacity values by `σ : Equiv.Perm (Fin n)`,
    leaving the rest unchanged. -/
def permuteAlpha (a : ℕ → ℝ) (n : ℕ) (σ : Equiv.Perm (Fin n)) : ℕ → ℝ :=
  fun i => if h : i < n then a (σ ⟨i, h⟩) else a i

/-- The product `∏ i in range n, (1 - a (σ i))` equals `∏ i in range n, (1 - a i)` for any
    permutation `σ` of `Fin n`. This follows from commutativity of multiplication. -/
lemma prod_one_sub_perm (a : ℕ → ℝ) (n : ℕ) (σ : Equiv.Perm (Fin n)) :
    ∏ i ∈ range n, (1 - permuteAlpha a n σ i) = ∏ i ∈ range n, (1 - a i) := by
  rw [← Fin.prod_univ_eq_prod_range, ← Fin.prod_univ_eq_prod_range]
  have : ∀ i : Fin n, (1 - permuteAlpha a n σ ↑i) = (1 - a ↑(σ i)) := by
    intro ⟨i, hi⟩; simp [permuteAlpha, hi]
  simp_rw [this]
  exact Equiv.Perm.prod_comp σ Finset.univ (fun i => 1 - a ↑i) (by simp)

/-- **Main theorem (H4):** The total rendering weight `∑ w(a, i)` is invariant under any
    permutation of the opacity values.

    By the telescoping identity, `∑ w(a, i) = 1 - ∏(1 - aᵢ)`. Since finite products are
    invariant under permutation (commutativity of multiplication), the total weight is
    the same regardless of rendering order. -/
theorem total_weight_perm_invariant (a : ℕ → ℝ) (n : ℕ) (σ : Equiv.Perm (Fin n)) :
    ∑ i ∈ range n, w (permuteAlpha a n σ) i =
    ∑ i ∈ range n, w a i := by
  rw [alpha_compositing_sum, alpha_compositing_sum]
  congr 1
  rw [T_eq_prod, T_eq_prod]
  exact prod_one_sub_perm a n σ

/-! ## Corollary: Two-element symmetry -/

/-- The total weight for two elements expressed via compositing weights equals
    `a₁ + a₂ - a₁ * a₂`. -/
theorem two_element_total_weight (a₁ a₂ : ℝ) :
    let a : ℕ → ℝ := fun | 0 => a₁ | 1 => a₂ | _ => 0
    ∑ i ∈ range 2, w a i = a₁ + a₂ - a₁ * a₂ := by
  simp [Finset.sum_range_succ, w, T]
  ring

/-- For two elements with opacities `a₁` and `a₂`, the total weight
    `a₁ + a₂ - a₁ * a₂` is symmetric: swapping the two elements doesn't change the total. -/
theorem two_element_total_weight_comm (a₁ a₂ : ℝ) :
    a₁ + a₂ - a₁ * a₂ = a₂ + a₁ - a₂ * a₁ := by ring

end
