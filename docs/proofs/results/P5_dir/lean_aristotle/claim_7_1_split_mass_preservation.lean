import Mathlib

/-!
# Gaussian Split Preserves Total Mass When Opacity Is Halved

We prove that splitting a single weighted Gaussian component into two equal-weight
components doubles the total integrated mass, and therefore the opacity (weight) of
each child must be halved to preserve total mass.

## Setup

Consider a Gaussian mixture component with:
- Weight (opacity): `α`
- Mean: `μ`
- Covariance: `Σ`

The total integrated weighted density is `α · ∫ N(x|μ, Σ) dx = α · 1 = α`.

After splitting into two Gaussians `N(μ + ε·v, Σ/4)` and `N(μ - ε·v, Σ/4)`,
each with weight `α`, the total becomes `α · 1 + α · 1 = 2α`.

To preserve total mass, each child should have weight `α/2`.
-/

open MeasureTheory

set_option maxHeartbeats 400000

/-! ### Core theorems about mass preservation -/

/-
The total mass after a naive split (keeping weight α for each child) is 2α,
    given that each component's PDF integrates to 1.
-/
theorem mass_after_naive_split (α : ℝ) (int_parent int_child1 int_child2 : ℝ)
    (_h_parent : int_parent = 1) (h_child1 : int_child1 = 1) (h_child2 : int_child2 = 1) :
    α * int_child1 + α * int_child2 = 2 * α := by
  grobner

/-
The total mass before the split is α.
-/
theorem mass_before_split (α : ℝ) (int_parent : ℝ) (h_parent : int_parent = 1) :
    α * int_parent = α := by
  rw [ h_parent, mul_one ]

/-
The naive split doubles the total mass.
-/
theorem naive_split_doubles_mass (α : ℝ) (int_parent int_child1 int_child2 : ℝ)
    (h_parent : int_parent = 1) (h_child1 : int_child1 = 1) (h_child2 : int_child2 = 1) :
    α * int_child1 + α * int_child2 = 2 * (α * int_parent) := by
  grind

/-
Halving the opacity of each child preserves total mass.
-/
theorem halved_opacity_preserves_mass (α : ℝ) (int_parent int_child1 int_child2 : ℝ)
    (h_parent : int_parent = 1) (h_child1 : int_child1 = 1) (h_child2 : int_child2 = 1) :
    (α / 2) * int_child1 + (α / 2) * int_child2 = α * int_parent := by
  rw [ h_child1, h_child2, h_parent ] ; ring

/-
The required new opacity for each child is α/2. This is because each child
    integrates to 1, so to have total mass α we need 2 · α_new = α, i.e. α_new = α/2.
-/
theorem opacity_correction (α α_new : ℝ) (int_child1 int_child2 : ℝ)
    (h_child1 : int_child1 = 1) (h_child2 : int_child2 = 1)
    (h_preserve : α_new * int_child1 + α_new * int_child2 = α) :
    α_new = α / 2 := by
  rw [h_child1, h_child2] at h_preserve
  linarith

/-
Combining everything: the split doubles the mass (from α to 2α),
    and halving the opacity restores it.
-/
theorem split_mass_ratio (α : ℝ) (int_parent int_child1 int_child2 : ℝ)
    (h_parent : int_parent = 1) (h_child1 : int_child1 = 1) (h_child2 : int_child2 = 1) :
    (α * int_child1 + α * int_child2) / (α * int_parent) = 2 ∨ α = 0 := by
  grind