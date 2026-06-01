import Mathlib

/-!
# Mahalanobis Distance and Chi-Squared Distribution

We prove that if Z₁, …, Z_d are independent standard normal random variables,
then D = ∑ᵢ Zᵢ² satisfies E[D] = d and Var[D] = 2d.

This captures the essential content of the classical result that the Mahalanobis distance
D = (X - μ)ᵀ Σ⁻¹ (X - μ) for X ~ N(μ, Σ) follows a χ²(d) distribution.
The key insight is that Σ = LLᵀ (Cholesky decomposition), so Z = L⁻¹(X - μ) ~ N(0, I),
and D = ZᵀZ = ∑ᵢ Zᵢ². Since the Zᵢ are iid N(0,1), D is by definition χ²(d).

## Main results

- `integral_sq_stdNormal`: E[Z²] = 1 for Z ~ N(0,1)
- `integral_fourth_pow_stdNormal`: E[Z⁴] = 3 for Z ~ N(0,1)
- `variance_sq_stdNormal`: Var[Z²] = 2 for Z ~ N(0,1)
- `chi_squared_mean`: E[∑ᵢ Zᵢ²] = d
- `chi_squared_variance`: Var[∑ᵢ Zᵢ²] = 2d
-/

open MeasureTheory ProbabilityTheory MeasureTheory.Measure
open scoped ENNReal NNReal BigOperators

noncomputable section

namespace MahalanobisChiSquared

/-- The standard normal distribution on ℝ. -/
abbrev stdNormal : Measure ℝ := gaussianReal 0 1

/-! ### Moment computations for the standard normal distribution -/

/-- E[Z] = 0 for Z ~ N(0,1). -/
theorem integral_id_stdNormal : ∫ x, x ∂stdNormal = 0 :=
  integral_id_gaussianReal

/-- Var[Z] = 1 for Z ~ N(0,1). -/
theorem variance_id_stdNormal : variance id stdNormal = 1 := by
  rw [variance_id_gaussianReal]
  simp

/-
E[Z²] = 1 for Z ~ N(0,1).
This follows from Var[Z] = E[Z²] - (E[Z])² = 1, and E[Z] = 0.
-/
theorem integral_sq_stdNormal : ∫ x, x ^ 2 ∂stdNormal = 1 := by
  have := @variance_id_stdNormal;
  rw [ ← this, ProbabilityTheory.variance, ProbabilityTheory.evariance_eq_lintegral_ofReal ];
  rw [ ← MeasureTheory.integral_eq_lintegral_of_nonneg_ae ];
  · simp +decide [ integral_id_stdNormal ];
  · exact Filter.Eventually.of_forall fun x => sq_nonneg _;
  · exact Continuous.aestronglyMeasurable ( by continuity )

/-
E[Z⁴] = 3 for Z ~ N(0,1). The fourth central moment of the standard normal.
-/
theorem integral_fourth_pow_stdNormal : ∫ x, x ^ 4 ∂stdNormal = 3 := by
  -- The fourth moment of the standard normal distribution is given by the integral of $x^4$ times the normal density function.
  have h_fourth_moment : ∫ x : ℝ, x^4 * (Real.exp (-x^2 / 2)) / Real.sqrt (2 * Real.pi) = 3 := by
    -- We'll use the fact that $\int_{-\infty}^{\infty} x^4 e^{-x^2/2} \, dx = 3\sqrt{2\pi}$.
    have h_gauss_moment : ∫ x : ℝ, x^4 * Real.exp (-x^2 / 2) = 3 * Real.sqrt (2 * Real.pi) := by
      have := @integral_rpow_mul_exp_neg_mul_rpow;
      have h_gauss_moment : ∫ x in Set.Ioi 0, x^4 * Real.exp (-x^2 / 2) = 3 * Real.sqrt (2 * Real.pi) / 2 := by
        convert @this 2 4 ( 1 / 2 ) ( by norm_num ) ( by norm_num ) ( by norm_num ) using 1 <;> norm_num [ div_eq_inv_mul ];
        rw [ show ( 5 / 2 : ℝ ) = 3 / 2 + 1 by norm_num, Real.Gamma_add_one ( by norm_num ), show ( 3 / 2 : ℝ ) = 1 / 2 + 1 by norm_num, Real.Gamma_add_one ( by norm_num ), Real.Gamma_one_half_eq ] ; ring ; norm_num;
        rw [ show ( - ( 5 / 2 : ℝ ) ) = -2 - 1 / 2 by norm_num, Real.rpow_sub ] <;> norm_num ; ring;
        norm_num [ ← Real.sqrt_eq_rpow ] ; ring;
      -- Since the integrand is even, we can double the integral over the positive half-line.
      have h_even : ∫ x in Set.Iic 0, x^4 * Real.exp (-x^2 / 2) = 3 * Real.sqrt (2 * Real.pi) / 2 := by
        rw [ ← h_gauss_moment, ← neg_zero, ← integral_comp_neg_Iic ] ; norm_num;
      convert congr_arg₂ ( · + · ) h_even h_gauss_moment using 1;
      · rw [ ← MeasureTheory.setIntegral_union ] <;> norm_num;
        · exact ( by contrapose! h_even; rw [ MeasureTheory.integral_undef h_even ] ; positivity );
        · exact ( by contrapose! h_gauss_moment; rw [ MeasureTheory.integral_undef h_gauss_moment ] ; positivity );
      · ring;
    rw [ MeasureTheory.integral_div, h_gauss_moment, mul_div_cancel_right₀ _ ( by positivity ) ];
  convert h_fourth_moment using 1;
  rw [ stdNormal, MeasureTheory.integral_eq_lintegral_of_nonneg_ae ];
  · rw [ MeasureTheory.integral_eq_lintegral_of_nonneg_ae ];
    · norm_num [ gaussianReal ];
      rw [ MeasureTheory.lintegral_withDensity_eq_lintegral_mul ] <;> norm_num [ gaussianPDF ];
      · norm_num [ gaussianPDFReal, mul_comm ];
        exact congr_arg _ ( MeasureTheory.lintegral_congr fun x => by rw [ ← ENNReal.ofReal_mul ( by positivity ) ] ; ring );
      · fun_prop;
      · exact Measurable.ennreal_ofReal ( measurable_id.pow_const _ );
    · exact Filter.Eventually.of_forall fun x => by positivity;
    · exact Continuous.aestronglyMeasurable ( by continuity );
  · exact Filter.Eventually.of_forall fun x => by positivity;
  · exact Continuous.aestronglyMeasurable ( by continuity )

/-
Var[Z²] = E[Z⁴] - (E[Z²])² = 3 - 1 = 2 for Z ~ N(0,1).
-/
theorem variance_sq_stdNormal : variance (· ^ 2) stdNormal = 2 := by
  -- We'll use that Z is a standard normal random variable to simplify the expression.
  have hZ : ∫ x, x ^ 2 ∂stdNormal = 1 ∧ ∫ x, x ^ 4 ∂stdNormal = 3 := by
    exact ⟨ integral_sq_stdNormal, integral_fourth_pow_stdNormal ⟩;
  rw [ ProbabilityTheory.variance, ProbabilityTheory.evariance_eq_lintegral_ofReal, ← MeasureTheory.integral_eq_lintegral_of_nonneg_ae ];
  · ring_nf;
    rw [ MeasureTheory.integral_add, MeasureTheory.integral_add ] <;> norm_num [ MeasureTheory.integral_neg, MeasureTheory.integral_const_mul, MeasureTheory.integral_mul_const, hZ ] ; ring;
    · exact MeasureTheory.Integrable.neg ( MeasureTheory.Integrable.mul_const ( by exact ( by contrapose! hZ; rw [ MeasureTheory.integral_undef hZ ] at *; norm_num at * ) ) _ );
    · exact ( by contrapose! hZ; rw [ MeasureTheory.integral_undef hZ ] ; norm_num );
    · refine' MeasureTheory.Integrable.add _ _;
      · exact MeasureTheory.Integrable.neg ( MeasureTheory.Integrable.mul_const ( by exact ( by contrapose! hZ; rw [ MeasureTheory.integral_undef hZ ] at *; norm_num at * ) ) _ );
      · exact ( by contrapose! hZ; rw [ MeasureTheory.integral_undef hZ ] ; norm_num );
  · exact Filter.Eventually.of_forall fun x => sq_nonneg _;
  · exact Continuous.aestronglyMeasurable ( by continuity )

/-! ### Main results: E[D] = d and Var[D] = 2d

We work with an abstract probability space Ω and independent random variables
Z_i : Ω → ℝ, each distributed as N(0,1). We define D = ∑ᵢ (Z_i)² and prove
E[D] = d and Var[D] = 2d.
-/

section AbstractProbabilitySpace

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {d : ℕ} {Z : Fin d → Ω → ℝ}

/-
Each Z_i has mean 0.
-/
omit [IsProbabilityMeasure μ] in
theorem mean_Z_eq_zero (i : Fin d) (hm : ∀ i, Measurable (Z i))
    (hd : ∀ i, Measure.map (Z i) μ = stdNormal) :
    ∫ ω, Z i ω ∂μ = 0 := by
  convert integral_id_stdNormal using 1;
  rw [ ← hd i, MeasureTheory.integral_map ];
  · exact hm i |> Measurable.aemeasurable;
  · exact measurable_id.aestronglyMeasurable

/-
Each Z_i² has expectation 1.
-/
omit [IsProbabilityMeasure μ] in
theorem integral_sq_Z (i : Fin d) (hm : ∀ i, Measurable (Z i))
    (hd : ∀ i, Measure.map (Z i) μ = stdNormal) :
    ∫ ω, (Z i ω) ^ 2 ∂μ = 1 := by
  rw [ ← integral_sq_stdNormal, ← hd i ];
  rw [ MeasureTheory.integral_map ];
  · exact hm i |> Measurable.aemeasurable;
  · exact Continuous.aestronglyMeasurable ( continuous_pow 2 )

/-
**Chi-squared mean**: E[∑ᵢ Zᵢ²] = d, where Z_i are iid N(0,1).
-/
omit [IsProbabilityMeasure μ] in
theorem chi_squared_mean (hm : ∀ i, Measurable (Z i))
    (hd : ∀ i, Measure.map (Z i) μ = stdNormal) :
    ∫ ω, (∑ i, (Z i ω) ^ 2) ∂μ = (d : ℝ) := by
  convert MeasureTheory.integral_finset_sum _ _;
  · rw [ Finset.sum_congr rfl fun i _ => integral_sq_Z i hm hd ] ; simp +decide;
  · intro i _;
    have h_integrable : MeasureTheory.Integrable (fun x => x ^ 2) (stdNormal) := by
      exact ( by have := integral_sq_stdNormal; exact ( by contrapose! this; rw [ MeasureTheory.integral_undef this ] ; norm_num ) );
    rw [ ← hd i ] at h_integrable;
    rwa [ MeasureTheory.integrable_map_measure ] at h_integrable;
    · exact h_integrable.1;
    · exact ( hm i ).aemeasurable

/-
Each Z_i² has variance 2 (follows from E[Z⁴]=3 and E[Z²]=1).
-/
omit [IsProbabilityMeasure μ] in
theorem variance_sq_Z (i : Fin d) (hm : ∀ i, Measurable (Z i))
    (hd : ∀ i, Measure.map (Z i) μ = stdNormal) :
    variance (fun ω => (Z i ω) ^ 2) μ = 2 := by
  convert variance_sq_stdNormal using 1;
  rw [ ← hd i ];
  rw [ ProbabilityTheory.variance, ProbabilityTheory.variance, ProbabilityTheory.evariance, ProbabilityTheory.evariance ];
  rw [ MeasureTheory.lintegral_map' ];
  · rw [ MeasureTheory.integral_map ];
    · exact hm i |> Measurable.aemeasurable;
    · exact Continuous.aestronglyMeasurable ( continuous_pow 2 );
  · fun_prop;
  · exact hm i |> Measurable.aemeasurable

/-
**Chi-squared variance**: Var[∑ᵢ Zᵢ²] = 2d, where Z_i are iid N(0,1).
-/
theorem chi_squared_variance
    (hm : ∀ i, Measurable (Z i))
    (hd : ∀ i, Measure.map (Z i) μ = stdNormal)
    (hi : iIndepFun (β := fun _ => ℝ) Z (μ := μ)) :
    variance (fun ω => ∑ i, (Z i ω) ^ 2) μ = 2 * (d : ℝ) := by
  -- By the properties of the variance, we can expand it as follows:
  have h_expand : ProbabilityTheory.variance (fun ω => ∑ i, (Z i ω) ^ 2) μ = ∑ i, ∑ j, ProbabilityTheory.covariance (fun ω => (Z i ω) ^ 2) (fun ω => (Z j ω) ^ 2) μ := by
    convert ProbabilityTheory.variance_sum _;
    · simp +decide [ Finset.sum_apply ];
    · infer_instance;
    · intro i;
      have h_var : ∫ ω, (Z i ω) ^ 4 ∂μ = 3 := by
        convert integral_fourth_pow_stdNormal using 1;
        rw [ ← hd i, MeasureTheory.integral_map ];
        · exact hm i |> Measurable.aemeasurable;
        · exact Continuous.aestronglyMeasurable ( by continuity );
      rw [ memLp_two_iff_integrable_sq ];
      · ring_nf;
        exact ( by contrapose! h_var; rw [ MeasureTheory.integral_undef h_var ] ; norm_num );
      · exact MeasureTheory.AEStronglyMeasurable.pow ( hm i |> Measurable.aestronglyMeasurable ) _;
  -- For i ≠ j, the Zᵢ² and Zⱼ² are independent (since the Zᵢ are), so Cov = 0 by IndepFun.covariance_eq_zero.
  have h_zero : ∀ i j, i ≠ j → ProbabilityTheory.covariance (fun ω => (Z i ω) ^ 2) (fun ω => (Z j ω) ^ 2) μ = 0 := by
    intro i j hij
    have h_indep : ProbabilityTheory.IndepFun (fun ω => (Z i ω) ^ 2) (fun ω => (Z j ω) ^ 2) μ := by
      exact hi.indepFun hij |> fun h => h.comp ( measurable_id'.pow_const 2 ) ( measurable_id'.pow_const 2 );
    apply_rules [ ProbabilityTheory.IndepFun.covariance_eq_zero ];
    · have h_integrable : ∫ ω, (Z i ω) ^ 4 ∂μ = 3 := by
        have h_integrable : ∫ ω, (Z i ω) ^ 4 ∂μ = ∫ x, x ^ 4 ∂(Measure.map (Z i) μ) := by
          rw [ MeasureTheory.integral_map ];
          · exact hm i |> Measurable.aemeasurable;
          · exact Continuous.aestronglyMeasurable ( by continuity );
        rw [ h_integrable, hd i, integral_fourth_pow_stdNormal ];
      rw [ memLp_two_iff_integrable_sq ];
      · exact ( by contrapose! h_integrable; rw [ MeasureTheory.integral_undef ( by simpa only [ ← pow_mul ] using h_integrable ) ] ; norm_num );
      · exact Measurable.aestronglyMeasurable ( hm i |> Measurable.pow_const <| 2 );
    · have h_integrable : MeasureTheory.Integrable (fun ω => (Z j ω) ^ 4) μ := by
        have h_integrable : MeasureTheory.Integrable (fun x => x ^ 4) (Measure.map (Z j) μ) := by
          rw [ hd ];
          have := @integral_fourth_pow_stdNormal;
          exact ( by contrapose! this; rw [ MeasureTheory.integral_undef this ] ; norm_num );
        rwa [ MeasureTheory.integrable_map_measure ] at h_integrable;
        · exact Continuous.aestronglyMeasurable ( by continuity );
        · exact hm j |> Measurable.aemeasurable;
      rw [ memLp_two_iff_integrable_sq ];
      · simpa only [ ← pow_mul ] using h_integrable;
      · exact ( hm j |> Measurable.pow_const <| 2 ) |> Measurable.aestronglyMeasurable;
  rw [ h_expand, Finset.sum_congr rfl fun i hi => Finset.sum_eq_single i ( fun j hj => by by_cases hij : i = j <;> aesop ) ( by aesop ) ];
  rw [ Finset.sum_congr rfl fun i _ => ProbabilityTheory.covariance_self _ ];
  · rw [ Finset.sum_congr rfl fun i _ => variance_sq_Z i hm hd ] ; simp +decide [ mul_comm ];
  · exact fun i _ => ( hm i |> Measurable.pow_const <| 2 ) |> Measurable.aemeasurable

end AbstractProbabilitySpace

end MahalanobisChiSquared