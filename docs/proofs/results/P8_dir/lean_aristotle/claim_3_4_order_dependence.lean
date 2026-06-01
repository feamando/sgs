import Mathlib

/-!
# Order-Dependence of Alpha Compositing

We prove by construction that the alpha-compositing equation used in Gaussian
splatting is order-dependent. Given two Gaussians with different features and
nonzero effective opacities, swapping the compositing order produces a different
result.

## Alpha Compositing Equation

For a sequence of Gaussians evaluated at query point `q`, the rendered color is:
  C(q) = Σᵢ αᵢ · Kᵢ(q) · fᵢ · Tᵢ(q)
where Tᵢ(q) = Πⱼ<ᵢ (1 - αⱼ · Kⱼ(q)) is the accumulated transmittance.

For two Gaussians with effective opacities `a = α·K(q)` and features `f`:
  composite(a₁, a₂, f₁, f₂) = a₁·f₁ + a₂·f₂·(1 - a₁)

## Main Results

- `composite_diff`: The explicit difference formula between the two orderings.
- `composite_order_dependent`: Compositing is order-dependent when both effective
  opacities are nonzero and features differ.
-/

set_option maxHeartbeats 800000

noncomputable section

/-- Alpha compositing of two Gaussians in front-to-back order.
    `a₁`, `a₂` are the effective opacities (αᵢ · Kᵢ(q)) and `f₁`, `f₂` are features. -/
def composite (a₁ a₂ f₁ f₂ : ℝ) : ℝ :=
  a₁ * f₁ + a₂ * f₂ * (1 - a₁)

/-- Compositing (a, b) gives: αₐKₐ·fₐ + αbKb·fb·(1 - αₐKₐ) -/
theorem composite_ab (aK_a aK_b f_a f_b : ℝ) :
    composite aK_a aK_b f_a f_b = aK_a * f_a + aK_b * f_b * (1 - aK_a) := by
  rfl

/-- Compositing (b, a) gives: αbKb·fb + αₐKₐ·fₐ·(1 - αbKb) -/
theorem composite_ba (aK_a aK_b f_a f_b : ℝ) :
    composite aK_b aK_a f_b f_a = aK_b * f_b + aK_a * f_a * (1 - aK_b) := by
  rfl

/-
**Key algebraic identity**: The difference between the two compositing orders
    factors as the product of both effective opacities times the feature difference.
-/
theorem composite_diff (a₁ a₂ f₁ f₂ : ℝ) :
    composite a₁ a₂ f₁ f₂ - composite a₂ a₁ f₂ f₁ = a₁ * a₂ * (f₁ - f₂) := by
  unfold composite; ring;

/-
**Order-dependence theorem**: If both Gaussians have nonzero effective opacity
    (αᵢ · Kᵢ > 0) and different features (fₐ ≠ f_b), then compositing in order
    (Gₐ, G_b) produces a different result than (G_b, Gₐ).
-/
theorem composite_order_dependent
    (aK_a aK_b f_a f_b : ℝ)
    (ha : aK_a > 0) (hb : aK_b > 0)
    (hf : f_a ≠ f_b) :
    composite aK_a aK_b f_a f_b ≠ composite aK_b aK_a f_b f_a := by
  exact fun h => hf <| mul_left_cancel₀ ( ne_of_gt <| mul_pos ha hb ) <| by linarith [ composite_diff aK_a aK_b f_a f_b ] ;

/-
The condition `αₐ·Kₐ ≠ α_b·K_b` provides additional asymmetry: not only do the
    composited colors differ, but also the accumulated transmittances differ after
    compositing the first Gaussian.
-/
theorem transmittance_asymmetry
    (aK_a aK_b : ℝ)
    (hne : aK_a ≠ aK_b) :
    (1 - aK_a) ≠ (1 - aK_b) := by
  aesop

/-
When `αₐ·Kₐ ≠ α_b·K_b`, the weight assigned to the second Gaussian differs
    between the two orderings, providing an explicit witness of order-dependence
    even at the level of individual Gaussian contributions.
-/
theorem weight_asymmetry
    (aK_a aK_b : ℝ)
    (_hne : aK_a * aK_b ≠ 0)
    (hne_ak : aK_a ≠ aK_b) :
    aK_b * (1 - aK_a) ≠ aK_a * (1 - aK_b) := by
  grind

/-
Explicit computation: the difference in second-Gaussian weights equals
    the difference in effective opacities.
-/
theorem weight_diff (aK_a aK_b : ℝ) :
    aK_b * (1 - aK_a) - aK_a * (1 - aK_b) = aK_b - aK_a := by
  ring

end