import Mathlib

open Matrix Finset BigOperators Real

noncomputable section

set_option maxHeartbeats 800000

/-!
# Positive Definiteness of the Anisotropic Gaussian Kernel

We prove that for any symmetric positive definite matrix `M`, the anisotropic Gaussian kernel
`K(x, y) = exp(-1/2 * (x - y)ᵀ M (x - y))` is positive definite in the Mercer sense.
That is, for any finite collection of points, the resulting Gram matrix is positive semi-definite.

## Proof strategy

1. Show the Hadamard product of PSD matrices is PSD (Schur product theorem).
2. Show the 1D Gaussian kernel gives a PSD Gram matrix (power series argument).
3. Show the isotropic Gaussian kernel is PSD (product of 1D kernels + Schur).
4. Reduce the anisotropic case to the isotropic case via `M = Bᵀ B`.
-/

variable {d n : ℕ}

/-- The Gram matrix of the anisotropic Gaussian kernel. -/
def gaussianGramMatrix (M : Matrix (Fin d) (Fin d) ℝ)
    (pts : Fin n → (Fin d → ℝ)) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of fun i j =>
    exp (-1/2 * dotProduct (pts i - pts j) (M.mulVec (pts i - pts j)))

/-
============================================================================
Part 1: Schur Product Theorem
============================================================================

The Hadamard (entrywise) product of two positive semi-definite real matrices
is positive semi-definite (Schur product theorem).

Proof: Since `A` is PSD, there exists `C` such that `A = Cᵀ C`, i.e.,
`A i j = ∑ k, C k i * C k j`. Then
  `∑ i j, v i * (A i j * B i j) * v j`
  `= ∑ k, ∑ i j, (v i * C k i) * B i j * (v j * C k j)`
  `= ∑ k, wₖᵀ B wₖ ≥ 0`
where `wₖ i = v i * C k i`.
-/
theorem schur_product_theorem
    (A B : Matrix (Fin n) (Fin n) ℝ) (hA : A.PosSemidef) (hB : B.PosSemidef) :
    (Matrix.of fun i j => A i j * B i j).PosSemidef := by
  -- By definition of positive semi-definite matrices, we know that their eigenvalues are non-negative.
  have h_eigenvalues : ∀ (M : Matrix (Fin n) (Fin n) ℝ), M.PosSemidef → ∃ (C : Matrix (Fin n) (Fin n) ℝ), M = C.transpose * C := by
    intro M hM
    have h_eigenvalues : ∃ (C : Matrix (Fin n) (Fin n) ℝ), M = C.transpose * C := by
      have h_pos_def : Matrix.PosSemidef M := hM
      have := Matrix.posSemidef_iff_eq_conjTranspose_mul_self.mp h_pos_def; aesop;
    exact h_eigenvalues;
  -- Let $C$ and $D$ be matrices such that $A = C^\top C$ and $B = D^\top D$.
  obtain ⟨C, hC⟩ := h_eigenvalues A hA
  obtain ⟨D, hD⟩ := h_eigenvalues B hB;
  -- By definition of Hadamard product, we have $(A \circ B)_{ij} = (C^\top C)_{ij} (D^\top D)_{ij} = \sum_{k} C_{ki} C_{kj} \sum_{l} D_{li} D_{lj}$.
  have h_hadamard : ∀ i j, (A i j) * (B i j) = ∑ k, ∑ l, (C k i * D l i) * (C k j * D l j) := by
    simp +decide [ *, Matrix.mul_apply, mul_assoc, mul_comm, mul_left_comm, Finset.mul_sum _ _ _, Finset.sum_mul ];
    exact fun i j => Finset.sum_comm.trans ( Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by ring );
  -- Let $E$ be the matrix with entries $E_{(k, l), i} = C_{ki} D_{li}$.
  set E : Matrix (Fin n × Fin n) (Fin n) ℝ := fun p i => C p.1 i * D p.2 i;
  -- Then $A \circ B = E^\top E$.
  have h_hadamard_eq : (fun i j => (A i j) * (B i j)) = E.transpose * E := by
    ext i j; simp +decide [ h_hadamard, Matrix.mul_apply ] ; ring;
    rw [ ← Finset.sum_product' ] ; congr ; ext ; ring!;
  convert Matrix.posSemidef_conjTranspose_mul_self E using 1

/-
============================================================================
Part 2: 1D Gaussian kernel is PSD
============================================================================

For any real numbers, the matrix with entries `exp(a_i * a_j)` is PSD.

Proof: `exp(a_i * a_j) = ∑ k, (a_i * a_j)^k / k! = ∑ k, (a_i^k * a_j^k) / k!`.
So `∑ i j, c_i * c_j * exp(a_i * a_j) = ∑ k, (1/k!) * (∑ i, c_i * a_i^k)² ≥ 0`.
The matrix is the entrywise limit of partial sums, each of which is PSD.
-/
theorem exp_mul_gram_posSemidef (a : Fin n → ℝ) :
    (Matrix.of fun i j : Fin n => exp (a i * a j)).PosSemidef := by
  refine' ⟨ _, fun x => _ ⟩;
  · ext i j; simp +decide [ mul_comm ] ;
  · -- By Fubini's theorem, we can interchange the order of summation.
    have h_fubini : ∑ i : Fin n, ∑ j : Fin n, (x i * x j * Real.exp ((a i) * (a j))) = ∑' k : ℕ, (1 / Nat.factorial k) * (∑ i : Fin n, x i * (a i) ^ k) * (∑ j : Fin n, x j * (a j) ^ k) := by
      have h_fubini : ∑ i : Fin n, ∑ j : Fin n, (x i * x j * (∑' k : ℕ, ((a i) * (a j)) ^ k / Nat.factorial k)) = ∑' k : ℕ, (∑ i : Fin n, ∑ j : Fin n, x i * x j * ((a i) * (a j)) ^ k / Nat.factorial k) := by
        have h_fubini : ∀ {f : Fin n → Fin n → ℕ → ℝ}, (∀ i j, Summable (fun k => f i j k)) → ∑ i, ∑ j, ∑' k, f i j k = ∑' k, ∑ i, ∑ j, f i j k := by
          intro f hf;
          have h_fubini : ∀ {f : Fin n → Fin n → ℕ → ℝ}, (∀ i j, Summable (fun k => f i j k)) → ∑ i, ∑ j, ∑' k, f i j k = ∑' k, ∑ i, ∑ j, f i j k := by
            intros f hf
            have h_summable : Summable (fun k => ∑ i, ∑ j, f i j k) := by
              exact summable_sum fun i _ => summable_sum fun j _ => hf i j
            have h_fubini : ∀ N : ℕ, ∑ i, ∑ j, ∑ k ∈ Finset.range N, f i j k = ∑ k ∈ Finset.range N, ∑ i, ∑ j, f i j k := by
              exact?
            have h_fubini : Filter.Tendsto (fun N => ∑ i, ∑ j, ∑ k ∈ Finset.range N, f i j k) Filter.atTop (nhds (∑ i, ∑ j, ∑' k, f i j k)) := by
              exact tendsto_finset_sum _ fun i _ => tendsto_finset_sum _ fun j _ => ( hf i j |> Summable.hasSum |> HasSum.tendsto_sum_nat );
            exact tendsto_nhds_unique h_fubini ( by simpa only [ * ] using h_summable.hasSum.tendsto_sum_nat );
          exact h_fubini hf;
        convert h_fubini _ using 3;
        · simp +decide only [mul_div_assoc, tsum_mul_left];
        · exact fun i j => Summable.of_norm <| by simpa [ mul_div_assoc ] using Summable.mul_left _ <| Real.summable_pow_div_factorial _;
      simp_all +decide [ Real.exp_eq_exp_ℝ, NormedSpace.exp_eq_tsum_div, mul_pow, mul_assoc, mul_comm, mul_left_comm, Finset.mul_sum _ _ _, Finset.sum_mul ];
      exact tsum_congr fun k => Finset.sum_congr rfl fun i hi => Finset.sum_congr rfl fun j hj => by ring;
    simp_all +decide [ mul_assoc, mul_comm, mul_left_comm, Finsupp.sum_fintype ];
    exact tsum_nonneg fun k => mul_nonneg ( inv_nonneg.2 ( Nat.cast_nonneg _ ) ) ( mul_self_nonneg _ )

/-
The 1D Gaussian kernel matrix `exp(-(a_i - a_j)²/2)` is PSD.

Proof: `(a_i - a_j)² = a_i² - 2 a_i a_j + a_j²`, so
`exp(-(a_i - a_j)²/2) = exp(-a_i²/2) * exp(a_i * a_j) * exp(-a_j²/2)`.
The matrix `exp(a_i * a_j)` is PSD by `exp_mul_gram_posSemidef`, and
multiplying by diagonal matrices `D_ii = exp(-a_i²/2)` preserves PSD.
-/
theorem gaussian_1d_posSemidef (a : Fin n → ℝ) :
    (Matrix.of fun i j : Fin n => exp (-(a i - a j) ^ 2 / 2)).PosSemidef := by
  by_contra h;
  -- We can write the matrix as $G = (diagonal d) * E * (diagonal d)$ where $d_i = \exp(-a_i^2/2)$ and $E_{ij} = \exp(a_i * a_j)$.
  set d : Fin n → ℝ := fun i => Real.exp (-(a i)^2 / 2)
  set E : Matrix (Fin n) (Fin n) ℝ := fun i j => Real.exp (a i * a j);
  convert h ?_;
  convert Matrix.PosSemidef.conjTranspose_mul_mul_same ( exp_mul_gram_posSemidef a ) ( Matrix.diagonal d ) using 1;
  ext i j; norm_num [ ← Real.exp_add ] ; ring;
  rw [ ← Real.exp_add, ← Real.exp_add ] ; ring

/-
============================================================================
Part 3: Isotropic Gaussian kernel is PSD
============================================================================

The isotropic Gaussian kernel matrix is PSD.

Proof: `‖x - y‖² = ∑ l, (x l - y l)²`, so
`exp(-‖x - y‖²/2) = ∏ l, exp(-(x l - y l)²/2)`.
Each factor gives a PSD matrix (by `gaussian_1d_posSemidef`), and
the Hadamard product of PSD matrices is PSD (by `schur_product_theorem`).
-/
theorem gaussian_isotropic_posSemidef (pts : Fin n → (Fin d → ℝ)) :
    (Matrix.of fun i j : Fin n =>
      exp (-dotProduct (pts i - pts j) (pts i - pts j) / 2)).PosSemidef := by
  -- By induction on $d$, we can show that the product of Gaussian kernels is positive semi-definite.
  have induction_step (d : ℕ) : ∀ (a : Fin d → (Fin n → ℝ)),
    (Matrix.of fun i j : Fin n => ∏ l : Fin d, Real.exp (-(a l i - a l j) ^ 2 / 2)).PosSemidef := by
      induction' d with d ih <;> simp_all +decide [ Fin.prod_univ_succ ];
      · constructor;
        · exact Matrix.ext fun i j => by simp +decide ;
        · norm_num [ Finsupp.sum_fintype, Matrix.mulVec, dotProduct ];
          exact fun x => by simpa only [ ← Finset.mul_sum _ _ _, ← Finset.sum_mul ] using mul_self_nonneg _;
      · intro a;
        convert schur_product_theorem _ _ _ _ using 1;
        · convert gaussian_1d_posSemidef ( fun i => a 0 i ) using 1;
        · exact ih _;
  convert induction_step _ fun l i => pts i l using 1;
  simp +decide [ sq, ← Real.exp_sum, ← Finset.mul_sum _ _ _, ← Finset.sum_div ];
  simp +decide [ sub_mul, mul_sub, dotProduct, Finset.sum_sub_distrib, sub_div ]

/-
============================================================================
Part 4: Main theorem
============================================================================

The anisotropic Gaussian kernel `K(x, y) = exp(-1/2 * (x - y)ᵀ M (x - y))`
gives a positive semi-definite Gram matrix for any positive definite matrix `M`.

Proof: Since `M` is positive definite, it is positive semi-definite, so there exists
a matrix `B` with `M = Bᴴ B`. Then `(x - y)ᵀ M (x - y) = ‖B(x - y)‖²`,
so the kernel becomes the isotropic Gaussian applied to the transformed points `Bx_i`.
The result follows from `gaussian_isotropic_posSemidef`.
-/
theorem gaussianGramMatrix_posSemidef
    (M : Matrix (Fin d) (Fin d) ℝ) (hM : M.PosDef)
    (pts : Fin n → (Fin d → ℝ)) :
    (gaussianGramMatrix M pts).PosSemidef := by
  obtain ⟨B, hB⟩ : ∃ B : Matrix (Fin d) (Fin d) ℝ, M = B.transpose * B := by
    have := Matrix.posSemidef_iff_eq_conjTranspose_mul_self.mp hM.posSemidef;
    aesop;
  unfold gaussianGramMatrix;
  -- Substitute $M = Bᵀ B$ into the Gram matrix.
  suffices h_subst : (Matrix.of fun i j => Real.exp (-1/2 * dotProduct (B.mulVec (pts i - pts j)) (B.mulVec (pts i - pts j)))).PosSemidef by
    convert h_subst using 3 ; simp +decide [ hB, Matrix.mul_assoc, Matrix.dotProduct_mulVec, Matrix.vecMul_mulVec ];
  convert gaussian_isotropic_posSemidef ( fun i => B.mulVec ( pts i ) ) using 1;
  norm_num [ Matrix.mulVec_sub, dotProduct_sub ];
  exact funext fun i => funext fun j => congr_arg Real.exp ( by ring )

end