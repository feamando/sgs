import Mathlib

open scoped BigOperators
open Real

set_option maxHeartbeats 8000000
set_option maxRecDepth 4000

set_option relaxedAutoImplicit false
set_option autoImplicit false

/-!
# Claim P8: Clone/Split Material Similarity Lipschitz Bound

When a Physical Gaussian G is cloned (position perturbed by ε) or split
(position offset along longest axis, scale reduced), the material similarity
M(G_a, G_b) between the children satisfies:

  M(G_a, G_b) ≥ 1 - L · ε

where L is the Lipschitz constant of M.

## Key Steps

1. `exp(-x)` is Lipschitz with constant 1 on `[0, ∞)`.
2. Cosine similarity is Lipschitz in its arguments.
3. A weighted sum of Lipschitz functions is Lipschitz with constant equal
   to the sum of weighted constants.
4. Therefore, small perturbations in position/scale produce small changes in M.

## Structure

We formalize the three key Lipschitz lemmas, then derive the main bound.
-/

/-! ## Part 1: exp(-x) is 1-Lipschitz on [0, ∞) -/

/-
The function `x ↦ exp(-x)` is 1-Lipschitz on `[0, ∞)`:
    `|exp(-x) - exp(-y)| ≤ |x - y|` for all `x, y ≥ 0`.
-/
theorem exp_neg_lipschitz (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
    |exp (-x) - exp (-y)| ≤ |x - y| := by
  -- By the Mean Value Theorem, there exists some $c$ between $x$ and $y$ such that $-\exp(-c) = \frac{\exp(-y) - \exp(-x)}{y - x}$.
  have h_mvt : ∀ {x y : ℝ}, 0 ≤ x → 0 ≤ y → x ≠ y → ∃ c ∈ Set.Ioo (min x y) (max x y), -Real.exp (-c) = (Real.exp (-y) - Real.exp (-x)) / (y - x) := by
    intros x y hx hy hxy
    have h_mvt : ∃ c ∈ Set.Ioo (min x y) (max x y), deriv (fun t => Real.exp (-t)) c = (Real.exp (-y) - Real.exp (-x)) / (y - x) := by
      cases le_total x y <;> simp_all +decide;
      · have := exists_deriv_eq_slope ( f := fun t => Real.exp ( -t ) ) ( show x < y from lt_of_le_of_ne ‹_› hxy );
        exact this ( Continuous.continuousOn <| by continuity ) ( Differentiable.differentiableOn <| by exact Differentiable.exp <| differentiable_id.neg );
      · have := exists_deriv_eq_slope ( f := fun t => Real.exp ( -t ) ) ( show y < x from lt_of_le_of_ne ‹_› ( Ne.symm hxy ) );
        exact this ( Continuous.continuousOn <| by continuity ) ( Differentiable.differentiableOn <| by exact Differentiable.exp <| differentiable_id.neg ) |> fun ⟨ c, hc₁, hc₂ ⟩ => ⟨ c, hc₁, by rw [ hc₂, ← neg_div_neg_eq ] ; ring ⟩;
    convert h_mvt using 3 ; norm_num [ Real.exp_ne_zero, Real.exp_neg, Real.differentiableAt_exp ];
    norm_num [ sq, neg_div ];
  cases eq_or_ne x y <;> simp_all +decide [ abs_div, div_le_iff₀ ];
  obtain ⟨ c, hc₁, hc₂ ⟩ := h_mvt hx hy ‹_› ; rw [ eq_div_iff ( sub_ne_zero_of_ne <| Ne.symm ‹_› ) ] at hc₂ ; cases abs_cases ( x - y ) <;> cases abs_cases ( Real.exp ( -x ) - Real.exp ( -y ) ) <;> nlinarith [ Real.exp_pos ( -c ), Real.exp_le_one_iff.mpr ( show -c ≤ 0 by cases hc₁.1 <;> cases hc₁.2 <;> linarith ) ]

/-! ## Part 2: Cosine similarity is Lipschitz

We model the *rescaled* cosine similarity `(cos θ + 1)/2` as a function of
the angle θ.  Since `d/dθ [(cos θ + 1)/2] = -sin θ / 2` and `|sin θ| ≤ 1`,
the function is `(1/2)`-Lipschitz.  More generally, if we view cosine
similarity as a function of the two unit vectors, it is 1-Lipschitz in
each argument.

For our purposes the key fact is: the rescaled cosine similarity
`f(u,v) = (⟨u,v⟩/‖u‖‖v‖ + 1)/2` changes by at most a bounded amount
when the arguments are perturbed.  We state a clean version for the
angle-parameterised form.
-/

/-
`(cos θ + 1)/2` is `(1/2)`-Lipschitz in θ.
-/
theorem rescaled_cos_lipschitz (θ₁ θ₂ : ℝ) :
    |(cos θ₁ + 1) / 2 - (cos θ₂ + 1) / 2| ≤ (1 / 2) * |θ₁ - θ₂| := by
  -- Use the identity $|\cos \theta_1 - \cos \theta_2| \leq |\theta_1 - \theta_2|$.
  have h_cos_diff : abs (Real.cos θ₁ - Real.cos θ₂) ≤ abs (θ₁ - θ₂) := by
    exact?;
  exact abs_le.mpr ⟨ by linarith [ abs_le.mp h_cos_diff ], by linarith [ abs_le.mp h_cos_diff ] ⟩

/-! ## Part 3: Weighted sum of Lipschitz functions is Lipschitz -/

/-
If each `fₖ` is `Lₖ`-Lipschitz and `wₖ ≥ 0`, then `∑ wₖ · fₖ` is
    `(∑ wₖ · Lₖ)`-Lipschitz. Here we state this for functions `ℝ → ℝ`.
-/
theorem weighted_sum_lipschitz
    {n : ℕ}
    (w : Fin n → ℝ)
    (L : Fin n → ℝ)
    (f : Fin n → ℝ → ℝ)
    (hw : ∀ i, 0 ≤ w i)
    (hL : ∀ i, 0 ≤ L i)
    (hf_lip : ∀ i, ∀ x y : ℝ, |f i x - f i y| ≤ L i * |x - y|)
    (x y : ℝ) :
    |∑ i, w i * f i x - ∑ i, w i * f i y| ≤ (∑ i, w i * L i) * |x - y| := by
  rw [ ← Finset.sum_sub_distrib, Finset.sum_mul _ _ _ ];
  exact le_trans ( Finset.abs_sum_le_sum_abs _ _ ) ( Finset.sum_le_sum fun i _ => by rw [ ← mul_sub ] ; exact abs_le.mpr ⟨ by nlinarith [ abs_le.mp ( hf_lip i x y ), hw i, hL i ], by nlinarith [ abs_le.mp ( hf_lip i x y ), hw i, hL i ] ⟩ )

/-! ## Part 4: Main Theorem – Clone/Split Similarity Bound

We model the material similarity `M` abstractly as a weighted combination
of component functions, each Lipschitz in a perturbation parameter `ε`.
When `ε = 0` every component equals 1 (self-similarity), so
`M(G, G) = ∑ wₖ · 1 = 1`.  When `ε > 0`, the Lipschitz bound gives
`M ≥ 1 - L · ε`.
-/

/-
**Claim P8 (abstract form).**

Given:
- weights `wₖ ≥ 0` summing to 1,
- component functions `fₖ(ε)` with `fₖ(0) = 1`,
- each `fₖ` is `Lₖ`-Lipschitz,

the material similarity `M(ε) = ∑ wₖ · fₖ(ε)` satisfies
`M(ε) ≥ 1 - (∑ wₖ · Lₖ) · ε` for every `ε ≥ 0`.
-/
theorem clone_split_similarity_bound
    {n : ℕ}
    (w : Fin n → ℝ)
    (L : Fin n → ℝ)
    (f : Fin n → ℝ → ℝ)
    (hw_nonneg : ∀ i, 0 ≤ w i)
    (hw_sum : ∑ i, w i = 1)
    (hf_self : ∀ i, f i 0 = 1)
    (hL_nonneg : ∀ i, 0 ≤ L i)
    (hf_lip : ∀ i, ∀ x y : ℝ, |f i x - f i y| ≤ L i * |x - y|)
    (ε : ℝ) (hε : 0 ≤ ε) :
    ∑ i, w i * f i ε ≥ 1 - (∑ i, w i * L i) * ε := by
  -- By the properties of the Lipschitz function and the fact that $f_i(0) = 1$, we can bound the terms $w_i * (f_i(ε) - f_i(0))$.
  have h_bound : ∀ i, w i * (f i ε - f i 0) ≥ w i * (-L i * ε) := by
    exact fun i => mul_le_mul_of_nonneg_left ( by cases abs_cases ( f i ε - f i 0 ) <;> cases abs_cases ( ε - 0 ) <;> nlinarith [ hf_lip i ε 0, hL_nonneg i ] ) ( hw_nonneg i );
  simp_all +decide [ mul_assoc, mul_comm, mul_left_comm, Finset.mul_sum _ _ _, Finset.sum_mul ];
  have := Finset.sum_le_sum fun i ( _ : i ∈ Finset.univ ) => h_bound i; simp_all +decide [ mul_sub, Finset.sum_add_distrib ] ;

/-
The self-similarity case: `M(0) = 1`.
-/
theorem material_similarity_self
    {n : ℕ}
    (w : Fin n → ℝ)
    (f : Fin n → ℝ → ℝ)
    (hw_sum : ∑ i, w i = 1)
    (hf_self : ∀ i, f i 0 = 1) :
    ∑ i, w i * f i 0 = 1 := by
  aesop

/-! ## Corollary: exp(-x) components satisfy the hypotheses -/

/-- `exp(-|·|)` evaluated at 0 gives 1, confirming the self-similarity
    hypothesis for exponential components. -/
theorem exp_component_self : exp (-|0|) = 1 := by
  simp [abs_zero, exp_zero]

/-
`exp(-|·|)` is 1-Lipschitz, confirming the Lipschitz hypothesis
    for exponential components.
-/
theorem exp_component_lipschitz (x y : ℝ) :
    |exp (-|x|) - exp (-|y|)| ≤ 1 * |x - y| := by
  -- Apply the triangle inequality to the exponential function.
  have h_triangle : abs (Real.exp (-|x|) - Real.exp (-|y|)) ≤ abs (|x| - |y|) := by
    convert exp_neg_lipschitz |x| |y| ( abs_nonneg x ) ( abs_nonneg y ) using 1;
  grind

#check @clone_split_similarity_bound
#check @material_similarity_self