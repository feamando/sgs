import Mathlib

/-!
# Alpha-Compositing Rendering Equation: Complete Gradient Flow

We formalize the alpha-compositing rendering equation and prove gradient flow properties.
-/

noncomputable section

open Finset Real

/-- Transmittance: product of (1 - α_j * K_j) for j < i -/
def T' (α K : ℕ → ℝ) (i : ℕ) : ℝ :=
  (Finset.range i).prod (fun j => 1 - α j * K j)

/-- Rendering weight: w_i = α_i * K_i * T_i -/
def w' (α K : ℕ → ℝ) (i : ℕ) : ℝ :=
  α i * K i * T' α K i

/-- Rendered value: Meaning = Σ_{i<N} w_i * f_i -/
def Meaning' (N : ℕ) (α K f : ℕ → ℝ) : ℝ :=
  (Finset.range N).sum (fun i => w' α K i * f i)

-- ============================================================================
-- Transmittance properties
-- ============================================================================

theorem T'_zero (α K : ℕ → ℝ) : T' α K 0 = 1 := by
  unfold T'; simp

theorem T'_succ (α K : ℕ → ℝ) (i : ℕ) :
    T' α K (i + 1) = T' α K i * (1 - α i * K i) := by
  unfold T'; rw [Finset.prod_range_succ]

-- ============================================================================
-- Part 1 (Scalar): d(Meaning)/d(f_i) = w_i
-- ============================================================================

/-- Meaning as a function of f_i, holding all else fixed -/
def Meaning_of_fi' (N : ℕ) (α K f : ℕ → ℝ) (i : ℕ) (fi : ℝ) : ℝ :=
  (Finset.range N).sum (fun j => w' α K j * (if j = i then fi else f j))

theorem hasDerivAt_Meaning_fi' (N : ℕ) (α K f : ℕ → ℝ) (i : ℕ) (hi : i < N) :
    HasDerivAt (Meaning_of_fi' N α K f i) (w' α K i) (f i) := by
  unfold Meaning_of_fi';
  simp +decide [ Finset.sum_ite, Finset.filter_eq', Finset.filter_ne' ];
  rw [ if_pos hi ] ; simpa using HasDerivAt.const_mul ( w' α K i ) ( hasDerivAt_id ( f i ) ) ;

theorem deriv_Meaning_fi_ne_zero' (N : ℕ) (α K f : ℕ → ℝ) (i : ℕ) (hi : i < N)
    (hw : w' α K i > 0) :
    deriv (Meaning_of_fi' N α K f i) (f i) ≠ 0 := by
  rw [ hasDerivAt_Meaning_fi' N α K f i hi |> HasDerivAt.deriv ] ; positivity

-- ============================================================================
-- Part 1 (Vector): d(Meaning)/d(f_i) = w_i • I
-- ============================================================================

/-- Vector Meaning as function of the i-th feature vector -/
def Meaning_vec_of_fi' {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    (N : ℕ) (α K : ℕ → ℝ) (f : ℕ → F) (i : ℕ) (fi : F) : F :=
  (Finset.range N).sum (fun j => w' α K j • (if j = i then fi else f j))

/-
d(Meaning)/d(f_i) = w_i • id (identity scaled by weight)
-/
theorem hasFDerivAt_Meaning_vec_fi' {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    (N : ℕ) (α K : ℕ → ℝ) (f : ℕ → F) (i : ℕ) (hi : i < N) :
    HasFDerivAt (Meaning_vec_of_fi' N α K f i)
      (w' α K i • ContinuousLinearMap.id ℝ F) (f i) := by
  rw [ hasFDerivAt_iff_isLittleO_nhds_zero ] at *;
  unfold Meaning_vec_of_fi';
  simp +decide [ ← Finset.sum_sub_distrib, Finset.sum_ite, Finset.filter_eq', Finset.filter_ne' ];
  aesop

/-
When w_i > 0, the Fréchet derivative w_i • I is nonzero
-/
theorem fderiv_Meaning_vec_fi_ne_zero' {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    [Nontrivial F]
    (N : ℕ) (α K : ℕ → ℝ) (f : ℕ → F) (i : ℕ) (hi : i < N)
    (hw : w' α K i > 0) :
    fderiv ℝ (Meaning_vec_of_fi' N α K f i) (f i) ≠ 0 := by
  obtain ⟨ v, hv ⟩ := exists_ne ( 0 : F );
  contrapose! hv;
  have := hasFDerivAt_Meaning_vec_fi' N α K f i hi;
  have := this.fderiv;
  simp_all +decide [ ContinuousLinearMap.ext_iff ];
  exact hv.resolve_left hw.ne' v

/-
============================================================================
Part 2: d(Meaning)/d(α_i) decomposition
============================================================================

When w_i > 0 and K_i > 0 (Gaussian kernel) and α_i ≥ 0, all three factors are positive
-/
theorem w'_pos_factors (α K : ℕ → ℝ) (i : ℕ)
    (hK : K i > 0) (hα : α i ≥ 0)
    (hw : w' α K i > 0) :
    α i > 0 ∧ K i > 0 ∧ T' α K i > 0 := by
  unfold w' at *;
  exact ⟨ lt_of_le_of_ne hα ( Ne.symm <| by aesop ), hK, lt_of_le_of_ne ( by nlinarith [ mul_nonneg hα hK.le ] ) ( Ne.symm <| by aesop ) ⟩

/-
The direct gradient coefficient K_i * T_i is positive when w_i > 0
    (with physical constraints: K_i > 0 from Gaussian, α_i ≥ 0 from opacity)
-/
theorem direct_coeff_pos' (α K : ℕ → ℝ) (i : ℕ)
    (hK : K i > 0) (hα : α i ≥ 0)
    (hw : w' α K i > 0) :
    K i * T' α K i > 0 := by
  exact mul_pos hK ( w'_pos_factors α K i hK hα hw |>.2.2 )

/-
T_i does not depend on α_i (only factors j < i appear)
-/
theorem T'_update_alpha (α K : ℕ → ℝ) (i : ℕ) (ai : ℝ) :
    T' (Function.update α i ai) K i = T' α K i := by
  exact Finset.prod_congr rfl fun j hj => by rw [ Function.update_of_ne ( by linarith [ Finset.mem_range.mp hj ] ) ] ;

/-- Meaning as a function of α_i using Function.update -/
def Meaning_of_alphai' (N : ℕ) (α K f : ℕ → ℝ) (i : ℕ) (ai : ℝ) : ℝ :=
  (Finset.range N).sum (fun j =>
    (Function.update α i ai) j * K j *
    T' (Function.update α i ai) K j * f j)

/-
The derivative of Meaning w.r.t. α_i decomposes as direct + indirect terms
-/
theorem hasDerivAt_Meaning_alphai_decomp' (N : ℕ) (α K f : ℕ → ℝ) (i : ℕ) (hi : i < N)
    (hαK : ∀ j, α j * K j ≠ 1) :
    ∃ indirect : ℝ,
    HasDerivAt (Meaning_of_alphai' N α K f i)
      (K i * T' α K i * f i + indirect) (α i) := by
  unfold Meaning_of_alphai';
  use deriv (fun ai => ∑ j ∈ Finset.range N, (Function.update α i ai) j * K j * T' (Function.update α i ai) K j * f j) (α i) - K i * T' α K i * f i;
  simp +zetaDelta at *;
  -- Since each term in the sum is differentiable, the sum itself is differentiable.
  have h_diff : ∀ j ∈ Finset.range N, DifferentiableAt ℝ (fun ai => (Function.update α i ai) j * K j * T' (Function.update α i ai) K j * f j) (α i) := by
    unfold T';
    intro j hj; norm_num [ Function.update_apply ] ;
    split_ifs;
    · apply_rules [ DifferentiableAt.mul, DifferentiableAt.prodMk, differentiableAt_id, differentiableAt_const ];
      induction' ( Finset.range j ) using Finset.induction <;> aesop;
    · apply_rules [ DifferentiableAt.mul, DifferentiableAt.prodMk ] <;> norm_num;
      induction' ( Finset.range j ) using Finset.induction <;> aesop;
  exact DifferentiableAt.fun_sum h_diff

-- ============================================================================
-- Part 3: d(Meaning)/d(μ_i) nonzero when q ≠ μ_i
-- ============================================================================

/-- 1D Gaussian kernel -/
def gaussianK' (q μ σ2 τ : ℝ) : ℝ :=
  exp (-(q - μ) ^ 2 / (2 * σ2 * τ))

/-
The Gaussian kernel is always positive
-/
theorem gaussianK'_pos (q μ σ2 τ : ℝ) : gaussianK' q μ σ2 τ > 0 := by
  exact Real.exp_pos _

/-
Derivative of Gaussian kernel w.r.t. μ
-/
theorem hasDerivAt_gaussianK'_mu (q μ σ2 τ : ℝ) (hσ : σ2 > 0) (hτ : τ > 0) :
    HasDerivAt (fun m => gaussianK' q m σ2 τ)
      (gaussianK' q μ σ2 τ * ((q - μ) / (σ2 * τ))) μ := by
  convert HasDerivAt.exp ( HasDerivAt.div_const ( HasDerivAt.neg ( HasDerivAt.comp μ ( hasDerivAt_pow 2 _ ) ( HasDerivAt.sub ( hasDerivAt_const _ _ ) ( hasDerivAt_id μ ) ) ) ) _ ) using 1 ; norm_num ; ring;
  unfold gaussianK'; ring;

/-
Derivative of Gaussian kernel w.r.t. μ is nonzero when q ≠ μ
-/
theorem deriv_gaussianK'_mu_ne_zero (q μ σ2 τ : ℝ) (hσ : σ2 > 0) (hτ : τ > 0)
    (hqμ : q ≠ μ) :
    deriv (fun m => gaussianK' q m σ2 τ) μ ≠ 0 := by
  unfold gaussianK';
  norm_num [ sub_sq, mul_comm ];
  grind

/-
The gradient factor α_i * T_i * ∂K_i/∂μ_i is nonzero when w_i > 0 and q ≠ μ_i
-/
theorem gradient_mu_factor_nonzero' (q μ σ2 τ αi Ti : ℝ)
    (hσ : σ2 > 0) (hτ : τ > 0) (hqμ : q ≠ μ)
    (hα : αi > 0) (hT : Ti > 0) :
    αi * Ti * deriv (fun m => gaussianK' q m σ2 τ) μ ≠ 0 := by
  rw [ show deriv ( fun m => gaussianK' q m σ2 τ ) μ = gaussianK' q μ σ2 τ * ( ( q - μ ) / ( σ2 * τ ) ) by exact HasDerivAt.deriv ( hasDerivAt_gaussianK'_mu q μ σ2 τ hσ hτ ) ] ; exact mul_ne_zero ( mul_ne_zero hα.ne' hT.ne' ) ( mul_ne_zero ( by exact ne_of_gt ( gaussianK'_pos q μ σ2 τ ) ) ( div_ne_zero ( sub_ne_zero_of_ne hqμ ) ( by positivity ) ) ) ;

/-
Combined: when w_i > 0 and q ≠ μ_i (with physical constraints), gradient to μ_i is nonzero
-/
theorem gradient_mu_nonzero_combined' (α K_vals : ℕ → ℝ)
    (q μ σ2 τ : ℝ) (i : ℕ)
    (hσ : σ2 > 0) (hτ : τ > 0) (hqμ : q ≠ μ)
    (hKi : K_vals i > 0) (hαi : α i ≥ 0)
    (hw : w' α K_vals i > 0) :
    α i * T' α K_vals i * deriv (fun m => gaussianK' q m σ2 τ) μ ≠ 0 := by
  have := w'_pos_factors α K_vals i hKi hαi hw;
  exact mul_ne_zero ( mul_ne_zero this.1.ne' this.2.2.ne' ) ( deriv_gaussianK'_mu_ne_zero q μ σ2 τ hσ hτ hqμ )

end