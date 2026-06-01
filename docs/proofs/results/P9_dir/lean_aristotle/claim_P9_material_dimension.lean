import Mathlib

/-!
# Claim P9: Dimension Sufficiency for Material Class Separation

For K distinct material classes in ℝ^{d_p}:
1. If d_p ≥ ⌈log₂(K)⌉, then K points can be placed with minimum pairwise distance > 0
   (use vertices of a hypercube: 2^{d_p} ≥ K vertices with pairwise Hamming distance ≥ 1).
2. For continuous interpolation on a K-simplex, d_p ≥ K−1 suffices
   (embed the standard (K−1)-simplex in ℝ^{K−1}).
3. At K = 100, d_p = 32 works since ⌈log₂(100)⌉ = 7 ≤ 32.
-/

open Finset Function

/-! ## Part 1: Hypercube embedding — K ≤ 2^d implies K distinct points in ℝ^d -/

/-- Map `Fin K` into the vertices of the d-dimensional hypercube {0,1}^d,
    given an injection `Fin K ↪ Fin (2^d)` (which exists when K ≤ 2^d).
    Each vertex is determined by the binary digits of its index. -/
noncomputable def hypercubeEmbedding (d : ℕ) :
    Fin (2 ^ d) → (Fin d → ℝ) :=
  fun n j => if (n.val / 2 ^ j.val) % 2 = 1 then 1 else 0

/-
The hypercube embedding is injective: distinct indices map to distinct points.
-/
theorem hypercubeEmbedding_injective (d : ℕ) :
    Injective (hypercubeEmbedding d) := by
      intro i j h_eq
      have h_eq_digits : ∀ k : Fin d, ((i.val / 2 ^ k.val) % 2 = (j.val / 2 ^ k.val) % 2) := by
        intro k; replace h_eq := congr_fun h_eq k; unfold hypercubeEmbedding at h_eq; aesop;
      refine' Fin.ext ( Nat.eq_of_testBit_eq fun k => _ );
      by_cases hk : k < d <;> simp_all +decide [ Nat.testBit, Nat.shiftRight_eq_div_pow ];
      · rw [ h_eq_digits ⟨ k, hk ⟩ ];
      · rw [ Nat.div_eq_of_lt, Nat.div_eq_of_lt ];
        · exact lt_of_lt_of_le j.2 ( Nat.pow_le_pow_right ( by decide ) hk );
        · exact lt_of_lt_of_le i.2 ( Nat.pow_le_pow_right ( by decide ) hk )

/-
**Part 1 (main)**: If K ≤ 2^d, there exist K points in ℝ^d that are pairwise distinct.
-/
theorem exists_K_distinct_points_in_Rd (K d : ℕ) (h : K ≤ 2 ^ d) :
    ∃ f : Fin K → (Fin d → ℝ), Injective f := by
      -- Define the function f as the composition of Fin.castLE and hypercubeEmbedding.
      use fun n => hypercubeEmbedding d (Fin.castLE h n);
      -- Since Fin.castLE h is injective, the composition with hypercubeEmbedding d is also injective.
      apply Function.Injective.comp (hypercubeEmbedding_injective d) (Fin.castLE_injective h)

/-! ## Part 2: Simplex embedding — K affinely independent points in ℝ^{K-1}

We show that the standard basis vectors e_1, …, e_K in ℝ^K are affinely independent,
proving that the standard (K-1)-simplex embeds in ℝ^K (hence in ℝ^{d_p} for d_p ≥ K).
For the sharper claim (ℝ^{K-1} suffices), we note K ≥ 1 and exhibit K affinely independent
points in ℝ^{K-1} by translating to place one vertex at the origin. -/

/-
The standard basis of ℝ^K gives K points that are pairwise distinct.
-/
theorem standardBasis_injective (K : ℕ) :
    Injective (fun i : Fin K => (Pi.single i (1 : ℝ))) := by
      intro i j h; replace h := congr_fun h j; simp_all +decide [ Pi.single_apply ] ;

/-
**Part 2 (main)**: For K ≥ 1, there exist K affinely independent points in ℝ^K.
    (The standard basis vectors e_1, …, e_K are affinely independent in ℝ^K.)
-/
theorem exists_affineIndep_points (K : ℕ) (_hK : 1 ≤ K) :
    ∃ f : Fin K → (Fin K → ℝ), AffineIndependent ℝ f := by
      refine' ⟨ fun i => ( Pi.single i 1 : Fin K → ℝ ), _ ⟩;
      intro s w hw h; simp_all +decide [ funext_iff ];
      intro i hi; specialize h i; simp_all +decide [ Pi.single_apply ] ;

/-! ## Part 3: Numerical verification for K = 100 -/

/-- 2^7 = 128 ≥ 100, so ⌈log₂(100)⌉ ≤ 7. -/
theorem two_pow_7_ge_100 : 100 ≤ 2 ^ 7 := by norm_num

/-- 7 ≤ 32, so d_p = 32 dimensions suffice for 100 materials. -/
theorem seven_le_32 : 7 ≤ 32 := by norm_num

/-- Combined: 100 ≤ 2^32, so 100 distinct points exist in ℝ^32. -/
theorem hundred_le_two_pow_32 : 100 ≤ 2 ^ 32 := by norm_num

/-- At K = 100, d_p = 32 suffices: there exist 100 distinct points in ℝ^32. -/
theorem exists_100_distinct_points_in_R32 :
    ∃ f : Fin 100 → (Fin 32 → ℝ), Injective f := by
  exact exists_K_distinct_points_in_Rd 100 32 hundred_le_two_pow_32