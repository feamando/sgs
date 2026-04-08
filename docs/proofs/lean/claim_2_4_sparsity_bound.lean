import Mathlib

open Real MeasureTheory Metric Finset
open scoped ENNReal NNReal BigOperators

noncomputable section

/-!
# Sparsity Bound for Gaussian Kernel Density Evaluation

We derive and prove a bound on the effective number of contributing Gaussians
in a kernel density evaluation.

## Setup
- `N` Gaussian distributions in `ℝ^d` with means `μ_1, ..., μ_N` in `B(0, R)`
- Isotropic covariance `σ²I`
- Kernel: `K_i(q) = exp(-‖q - μ_i‖² / (2σ²τ))`
- `S(q) = |{i : K_i(q) > ε}|` counts Gaussians with non-negligible contribution

## Main Results
1. **Threshold Radius Lemma**: `K_i(q) > ε ↔ ‖q - μ_i‖ < σ√(2τ ln(1/ε))`
2. **Volume Ratio**: `vol(B(r)) / vol(B(R)) = (r/R)^d`
3. **Sparsity Bound**: The contributing region has measure fraction `≤ (r*/R)^d`
4. **Asymptotic Sparsity**: For `d = 64, τ = 64, ε = 10⁻³, σ = 1, R ≫ 1`:
   `S(q)/N → 0`

The key insight is that in high dimensions, only Gaussians within a ball of radius
`r* = σ√(2τ ln(1/ε))` contribute, and the volume fraction of this ball relative
to the ambient ball `B(0,R)` shrinks as `(r*/R)^d`.
-/

variable {d : ℕ}

-- ============================================================================
-- Section 1: Definitions
-- ============================================================================

/-- The Gaussian kernel evaluation at query point `q` for a Gaussian centered at `mu`. -/
def gaussianKernel (sigma tau : ℝ) (q mu : EuclideanSpace ℝ (Fin d)) : ℝ :=
  Real.exp (-(‖q - mu‖ ^ 2) / (2 * sigma ^ 2 * tau))

/-- The threshold radius beyond which the kernel contribution is below ε. -/
def thresholdRadius (sigma tau epsilon : ℝ) : ℝ :=
  sigma * Real.sqrt (2 * tau * Real.log (1 / epsilon))

/-
============================================================================
Section 2: Threshold Radius Characterization
============================================================================

Auxiliary: the exponent manipulation for the threshold characterization.
-/
theorem exp_neg_div_gt_iff {a b : ℝ} (hb : 0 < b) {eps : ℝ} (heps : 0 < eps) :
    Real.exp (-a / b) > eps ↔ a / b < Real.log (1 / eps) := by
  rw [ gt_iff_lt, ← Real.log_lt_iff_lt_exp ( by positivity ) ];
  rw [ one_div, Real.log_inv ] ; constructor <;> intro h <;> ring_nf at * <;> linarith

/-
**Threshold Radius Lemma**: The Gaussian kernel `K(q, μ) = exp(-‖q-μ‖²/(2σ²τ))`
exceeds the threshold `ε` if and only if `μ` lies within the threshold radius
`r* = σ√(2τ ln(1/ε))` of the query point `q`.
-/
theorem kernel_exceeds_iff_within_radius
    {sigma tau epsilon : ℝ} {q mu : EuclideanSpace ℝ (Fin d)}
    (hsigma : 0 < sigma) (htau : 0 < tau) (heps_pos : 0 < epsilon) (heps_lt : epsilon < 1) :
    gaussianKernel sigma tau q mu > epsilon ↔
    ‖q - mu‖ < thresholdRadius sigma tau epsilon := by
  unfold gaussianKernel thresholdRadius;
  rw [ ← Real.exp_log heps_pos ];
  norm_num [ div_eq_mul_inv, mul_assoc, mul_comm, mul_left_comm, hsigma.le, htau.le ];
  field_simp;
  constructor <;> intro h;
  · rw [ ← Real.sqrt_sq hsigma.le ];
    rw [ ← Real.sqrt_mul <| by positivity ] ; exact Real.lt_sqrt_of_sq_lt <| by nlinarith;
  · nlinarith [ show 0 ≤ sigma * Real.sqrt ( - ( Real.log epsilon * tau * 2 ) ) by positivity, Real.mul_self_sqrt ( show 0 ≤ - ( Real.log epsilon * tau * 2 ) by nlinarith [ Real.log_le_sub_one_of_pos heps_pos, mul_pos htau heps_pos ] ), norm_nonneg ( q - mu ), mul_lt_mul_of_pos_left h ( show 0 < sigma by positivity ) ]

/-
============================================================================
Section 3: Volume Ratio for Balls in ℝ^d
============================================================================

**Volume Ratio Lemma**: In `ℝ^d`, the ratio of volumes of concentric balls
equals the ratio of radii raised to the `d`-th power.
-/
theorem volume_ball_ratio [NeZero d] (x : EuclideanSpace ℝ (Fin d))
    {r R : ℝ} (hr : 0 < r) (hR : 0 < R) :
    volume (ball x r) / volume (ball x R) = ENNReal.ofReal ((r / R) ^ (d : ℕ)) := by
  -- Use the volume formula for balls in Euclidean space.
  have h_volume : ∀ (r : ℝ), 0 < r → volume (ball x r) = ENNReal.ofReal (Real.sqrt (Real.pi) ^ d / Real.Gamma (d / 2 + 1)) * ENNReal.ofReal (r ^ d) := by
    intro r hr; rw [ EuclideanSpace.volume_ball ] ; ring;
    rw [ mul_comm, ENNReal.ofReal_pow hr.le ] ; norm_num;
  simp_all +decide [ mul_div_mul_right, ne_of_gt ];
  rw [ ENNReal.mul_div_mul_left _ _ ( by positivity ) ];
  · rw [ ← ENNReal.ofReal_div_of_pos ( by positivity ), div_pow ];
  · exact ENNReal.ofReal_ne_top

/-
============================================================================
Section 4: Measure Fraction Bound
============================================================================

The intersection of two balls has measure at most the smaller ball.
-/
theorem volume_inter_ball_le (q c : EuclideanSpace ℝ (Fin d)) (r R : ℝ) :
    volume (ball q r ∩ ball c R) ≤ volume (ball q r) := by
  exact MeasureTheory.measure_mono Set.inter_subset_left

/-
**Measure Fraction Bound**: The fraction of a ball `B(c, R)` that lies within
distance `r` of any point `q` is at most `(r/R)^d`.

This is the key geometric estimate: the "relevant" region (where kernels are
non-negligible) occupies only a `(r*/R)^d` fraction of the ambient ball.
-/
theorem measure_fraction_bound [NeZero d] (q c : EuclideanSpace ℝ (Fin d))
    {r R : ℝ} (hr : 0 < r) (hR : 0 < R) (hrR : r ≤ R) :
    volume (ball q r ∩ ball c R) / volume (ball c R) ≤
    ENNReal.ofReal ((r / R) ^ (d : ℕ)) := by
  -- We know that `ball q r ∩ ball c R ⊆ ball q r`
  have h_subset : ball q r ∩ ball c R ⊆ ball q r := by
    exact Set.inter_subset_left;
  refine' le_trans _ ( volume_ball_ratio q hr hR |> le_of_eq );
  apply_rules [ ENNReal.div_le_div ];
  · exact MeasureTheory.measure_mono h_subset;
  · rw [ ← MeasureTheory.measure_preimage_add_right ] ; norm_num;
    rw [ show q - ( q - c ) = c by abel1 ]

/-
============================================================================
Section 5: Main Sparsity Bound
============================================================================

**Main Sparsity Theorem**: The measure of the set of centers in `B(0, R)` whose
Gaussian kernel exceeds `ε` at query `q` is at most `(r*/R)^d` times the total
volume of `B(0, R)`, where `r* = σ√(2τ ln(1/ε))` is the threshold radius.

For `N` means uniformly distributed in `B(0, R)`, this implies:
  `E[S(q)] ≤ N · (σ√(2τ ln(1/ε)) / R)^d`
-/
theorem sparsity_bound [NeZero d]
    (q : EuclideanSpace ℝ (Fin d))
    {sigma tau epsilon R : ℝ}
    (hsigma : 0 < sigma) (htau : 0 < tau)
    (heps_pos : 0 < epsilon) (heps_lt : epsilon < 1) (hR : 0 < R)
    (hrR : thresholdRadius sigma tau epsilon ≤ R) :
    volume ({mu ∈ ball (0 : EuclideanSpace ℝ (Fin d)) R |
      gaussianKernel sigma tau q mu > epsilon}) /
    volume (ball (0 : EuclideanSpace ℝ (Fin d)) R) ≤
    ENNReal.ofReal ((thresholdRadius sigma tau epsilon / R) ^ (d : ℕ)) := by
  convert ( measure_fraction_bound q 0 _ _ hrR ) using 1;
  · congr! 2;
    ext; simp [kernel_exceeds_iff_within_radius hsigma htau heps_pos heps_lt];
    rw [ dist_eq_norm' ] ; tauto;
  · exact mul_pos hsigma ( Real.sqrt_pos.mpr ( mul_pos ( mul_pos two_pos htau ) ( Real.log_pos ( one_lt_one_div heps_pos heps_lt ) ) ) );
  · exact hR

/-
============================================================================
Section 6: Combinatorial Counting Bound
============================================================================

**Counting Bound**: For any finite collection of `N` means, the number of
Gaussians contributing above threshold ε at query q is at most the number of
means within the threshold radius r* of q.
-/
theorem contributing_count_le_ball_count
    {N : ℕ} {sigma tau epsilon : ℝ}
    {q : EuclideanSpace ℝ (Fin d)}
    (mus : Fin N → EuclideanSpace ℝ (Fin d))
    (hsigma : 0 < sigma) (htau : 0 < tau)
    (heps_pos : 0 < epsilon) (heps_lt : epsilon < 1) :
    (Finset.univ.filter (fun i => gaussianKernel sigma tau q (mus i) > epsilon)).card ≤
    (Finset.univ.filter (fun i => ‖q - mus i‖ < thresholdRadius sigma tau epsilon)).card := by
  grind +suggestions

/-
============================================================================
Section 7: Asymptotic Sparsity (Qualitative)
============================================================================

The volume fraction `(r*/R)^d → 0` as `R → ∞` for fixed `r* > 0` and `d ≥ 1`.
-/
theorem volume_fraction_tendsto_zero {rstar : ℝ} (hrstar : 0 < rstar) (hd : 0 < d) :
    Filter.Tendsto (fun R : ℝ => (rstar / R) ^ d)
    Filter.atTop (nhds 0) := by
  exact Filter.Tendsto.pow ( tendsto_const_nhds.div_atTop Filter.tendsto_id ) d |> fun h => h.trans ( by norm_num [ hd.ne' ] )

/-
============================================================================
Section 8: Numerical Verification
============================================================================

For the specific parameters d=64, τ=64, ε=10⁻³, σ=1,
the threshold radius is r* = √(128 · ln(1000)).
The bound (r*/R)^d vanishes as R → ∞, confirming sparsity.
-/
theorem numerical_sparsity_example :
    let sigma := (1 : ℝ)
    let tau := (64 : ℝ)
    let epsilon := (1 / 1000 : ℝ)
    let d_val := 64
    let rstar := thresholdRadius sigma tau epsilon
    -- r* = √(128 · ln(1000)) which is finite and positive
    0 < rstar ∧
    -- The volume fraction → 0 as R → ∞
    Filter.Tendsto (fun R : ℝ => (rstar / R) ^ d_val) Filter.atTop (nhds 0) := by
  unfold thresholdRadius; norm_num;
  exact ⟨ Real.log_pos ( by norm_num ), by exact le_trans ( Filter.Tendsto.pow ( tendsto_const_nhds.div_atTop Filter.tendsto_id ) _ ) ( by norm_num ) ⟩

/-
The threshold radius for σ=1, τ=64, ε=10⁻³ is at most 30.
This means for R ≥ 30, only a fraction ≤ (30/R)^64 of Gaussians contribute,
which is astronomically small for R ≫ 30.
-/
theorem threshold_radius_bounded :
    thresholdRadius 1 64 (1/1000 : ℝ) < 30 := by
  unfold thresholdRadius;
  rw [ ← lt_div_iff₀' ] <;> norm_num [ Real.sqrt_lt', Real.log_neg ];
  -- We know that $e^7 > 1000$, so taking the natural logarithm of both sides gives $7 > \log(1000)$.
  have h_exp : Real.exp 7 > 1000 := by
    have := Real.exp_one_gt_d9.le ; norm_num at * ; rw [ show Real.exp 7 = ( Real.exp 1 ) ^ 7 by rw [ ← Real.exp_nat_mul ] ; norm_num ] ; nlinarith [ pow_le_pow_left₀ ( by positivity ) this 7 ];
  rw [ ← Real.sqrt_mul <| by positivity ] ; exact Real.sqrt_lt' ( by positivity ) |>.2 <| by nlinarith [ Real.log_exp 7, Real.log_lt_log ( by positivity ) h_exp ] ;

end