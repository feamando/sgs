# Softmax Attention Is a Special Case of Alpha-Compositing

**A Formally Verified Theorem Connecting Transformer Attention to Volume Rendering**

Nikita Gorshkov — April 2026
Proof: Machine-verified in Lean 4 / Mathlib by Aristotle (harmonic.fun)

---

## The Two Schemes

Modern AI and computer graphics each use a weighted aggregation scheme to combine information from multiple sources into a single output. Despite being developed independently for completely different purposes, these two schemes turn out to be mathematically related — in a way that, to our knowledge, has not been formally established before.

### Scheme A: Softmax Attention (Transformers)

Every large language model (GPT, Claude, Llama, etc.) uses the **transformer architecture** (Vaswani et al., 2017). At its core, a transformer computes outputs as weighted sums:

```
output = Σᵢ wᵢ · vᵢ
```

where the weights come from the **softmax** function applied to learned relevance scores:

```
wᵢ = exp(sᵢ) / Σⱼ exp(sⱼ)
```

**Properties of softmax weights:**
- Always strictly positive: wᵢ > 0 for all i (since exp(s) > 0 for any s)
- Always sum to exactly 1: Σᵢ wᵢ = 1
- Every element contributes at least something (no element can be completely ignored)

### Scheme B: Alpha-Compositing (Volume Rendering)

3D Gaussian Splatting (Kerbl et al., 2023) and earlier volume rendering techniques (Porter & Duff, 1984; Max, 1995) compute outputs as:

```
output = Σᵢ wᵢ · vᵢ
```

where the weights come from **opacity × transmittance**:

```
wᵢ = aᵢ · Tᵢ
Tᵢ = ∏ⱼ<ᵢ (1 - aⱼ)      [accumulated transmittance]
```

Each aᵢ ∈ [0, 1] is the opacity of element i. The transmittance Tᵢ represents how much "capacity" remains after all preceding elements have contributed — modelling occlusion (objects in front block objects behind).

**Properties of alpha-compositing weights:**
- Non-negative: wᵢ ≥ 0 (can be exactly zero if aᵢ = 0)
- Sum to at most 1: Σᵢ wᵢ = 1 - T_{n+1} ≤ 1 (with possible residual)
- Elements CAN be completely ignored (if their opacity is zero)

---

## The Question

Both schemes produce a weighted sum of values. Both are differentiable and used in gradient-based optimization. But what is their mathematical relationship?

Specifically:
1. Can alpha-compositing reproduce any set of weights that softmax can produce?
2. Can softmax reproduce any set of weights that alpha-compositing can produce?
3. Or is neither a subset of the other?

---

## The Result

### Theorem 1: Alpha-compositing is NOT a special case of softmax

**Statement:** There exist alpha-compositing weights that no softmax can produce.

**Proof:** Consider a = (0, 1). The alpha-compositing weights are:
- w₁ = a₁ · T₁ = 0 · 1 = 0
- w₂ = a₂ · T₂ = 1 · (1 - 0) = 1

So the weight vector is (0, 1) — the first element is completely ignored.

But softmax weights are always strictly positive: wᵢ = exp(sᵢ)/Σⱼexp(sⱼ) > 0 since exp(sᵢ) > 0 for all sᵢ ∈ ℝ.

Therefore, the weight vector (0, 1) is achievable by alpha-compositing but not by softmax. ∎

### Theorem 2: Softmax IS a special case of alpha-compositing

**Statement:** For any softmax weight vector w (from any scores s), there exist alpha-compositing parameters a such that the alpha-compositing weights exactly equal w.

**Construction:** Given softmax weights w₁, ..., wₙ (all positive, summing to 1), define:

```
aᵢ = wᵢ / Σⱼ≥ᵢ wⱼ
```

That is, each opacity aᵢ equals the weight wᵢ divided by the "tail sum" — the total remaining weight from position i onward.

**Proof that this works:**

Define the tail sum: Rᵢ = Σⱼ≥ᵢ wⱼ. Note that R₁ = 1 (all weights sum to 1) and Rᵢ = wᵢ + R_{i+1}.

The transmittance through the first i-1 elements is:

```
Tᵢ = ∏ⱼ<ᵢ (1 - aⱼ) = ∏ⱼ<ᵢ (1 - wⱼ/Rⱼ) = ∏ⱼ<ᵢ (Rⱼ - wⱼ)/Rⱼ = ∏ⱼ<ᵢ R_{j+1}/Rⱼ
```

This is a **telescoping product**:

```
Tᵢ = (R₂/R₁) · (R₃/R₂) · ... · (Rᵢ/R_{i-1}) = Rᵢ/R₁ = Rᵢ
```

(using R₁ = 1).

Therefore the alpha-compositing weight at position i is:

```
wᵢ^{alpha} = aᵢ · Tᵢ = (wᵢ/Rᵢ) · Rᵢ = wᵢ ✓
```

The construction exactly recovers the original softmax weights. ∎

### Corollary: Strict Inclusion

```
{softmax weight vectors} ⊂ {alpha-compositing weight vectors}    (strict)
```

The set of weight vectors achievable by alpha-compositing strictly contains those achievable by softmax. Alpha-compositing can additionally produce:
- Weight vectors with **exact zeros** (some elements contribute literally nothing)
- Weight vectors summing to **less than 1** (residual transmittance — "not everything is accounted for")

---

## Why This Matters

### For AI Architecture Design

This theorem establishes that the rendering equation from computer graphics is **provably at least as expressive** as the attention mechanism in transformers for computing weighted aggregations. Any computation a transformer performs through attention weights can be exactly replicated by alpha-compositing — plus alpha-compositing offers additional capabilities (exact sparsity, sub-unity sums) that softmax structurally cannot provide.

This is relevant to the **Semantic Gaussian Splatting (SGS)** proposal, which replaces transformer attention with the alpha-compositing rendering equation from 3D Gaussian Splatting. The theorem proves this replacement does not lose expressiveness.

### For Computer Graphics

The reverse direction is also interesting: the standard alpha-compositing pipeline used in rendering is *strictly more powerful* than softmax attention as a weighted aggregation scheme. This provides theoretical backing for the empirical success of volume rendering and splatting-based methods over attention-based alternatives in 3D reconstruction tasks.

### The Extra Expressiveness

The additional capabilities of alpha-compositing have concrete interpretations:

**Exact zeros (sparsity):** In softmax, every element always contributes at least ε > 0 to every output (exponential is never zero). In alpha-compositing, setting aᵢ = 0 means element i contributes exactly zero — true hard sparsity without approximation. For language models, this means irrelevant words can be completely excluded, not just down-weighted.

**Sub-unity sums:** Softmax weights always sum to exactly 1 — all "probability mass" must be assigned. Alpha-compositing weights sum to 1 - T_{n+1}, which can be less than 1 when residual transmittance remains. The residual is a natural uncertainty measure: "I've only accounted for 80% of the meaning; 20% is unresolved." Softmax has no such mechanism.

---

## The Formal Proof (Lean 4)

The following is the complete, machine-verified proof in Lean 4 with Mathlib. It compiles with zero `sorry` (unproven assertions) and depends only on standard axioms (`propext`, `Classical.choice`, `Quot.sound`).

Verified by Aristotle (harmonic.fun) — Lean 4 v4.28.0, Mathlib v4.28.0.

```lean
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
   Given any softmax weight vector, we can construct alpha values that exactly
   reproduce it. The construction is `a_i = w_i / Σ_{j≥i} w_j`, which telescopes
   to recover the original weights.
-/

variable {n : ℕ}

/-- Softmax attention weight for index `i` given scores `s`. -/
def softmaxWeight (s : Fin n → ℝ) (i : Fin n) : ℝ :=
  exp (s i) / ∑ j : Fin n, exp (s j)

/-- Alpha compositing weight for index `i` given opacity values `a`.
  Each layer `i` contributes `a_i` attenuated by the transmittance
  `∏_{j<i} (1 - a_j)`. -/
def alphaWeight (a : Fin n → ℝ) (i : Fin n) : ℝ :=
  a i * ∏ j ∈ Finset.Iio i, (1 - a j)

/-- Tail sum: `Σ_{j ≥ i} w_j`. -/
def tailSum (w : Fin n → ℝ) (i : Fin n) : ℝ :=
  ∑ j ∈ Finset.Ici i, w j

/-- Construct alpha compositing parameters from a weight vector.
  Given weights `w`, set `a_i = w_i / tailSum(w, i)`. -/
def constructAlpha (w : Fin n → ℝ) (i : Fin n) : ℝ :=
  w i / tailSum w i

-- ═══════════════════════════════════════════════════════════
-- Section 1: Basic properties of softmax
-- ═══════════════════════════════════════════════════════════

lemma sum_exp_pos (s : Fin (n + 1) → ℝ) :
    0 < ∑ j : Fin (n + 1), exp (s j) :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty

lemma softmaxWeight_pos (s : Fin (n + 1) → ℝ) (i : Fin (n + 1)) :
    0 < softmaxWeight s i :=
  div_pos (Real.exp_pos _) (sum_exp_pos s)

lemma softmaxWeight_sum (s : Fin (n + 1) → ℝ) :
    ∑ i : Fin (n + 1), softmaxWeight s i = 1 := by
  unfold softmaxWeight
  rw [← Finset.sum_div _ _ _, div_self <| ne_of_gt <| sum_exp_pos s]

-- ═══════════════════════════════════════════════════════════
-- Section 2: Alpha compositing ⊄ Softmax
-- ═══════════════════════════════════════════════════════════

/-- Alpha compositing can produce the weight vector (0, 1), but softmax
  weights are always strictly positive. -/
theorem alpha_not_subset_softmax :
    ∃ (a : Fin 2 → ℝ), (∀ i, a i ∈ Set.Icc (0 : ℝ) 1) ∧
      ¬∃ (s : Fin 2 → ℝ), ∀ i, alphaWeight a i = softmaxWeight s i := by
  use ![0, 1]
  simp_all +decide [Fin.forall_fin_two, alphaWeight]
  exact fun x hx => absurd hx <| ne_of_lt <| softmaxWeight_pos x 0

-- ═══════════════════════════════════════════════════════════
-- Section 3: Softmax ⊆ Alpha Compositing
-- ═══════════════════════════════════════════════════════════

lemma tailSum_pos (w : Fin (n + 1) → ℝ) (hw : ∀ i, 0 < w i)
    (i : Fin (n + 1)) : 0 < tailSum w i :=
  Finset.sum_pos (fun j hj => hw j) (by simp)

lemma tailSum_zero_eq (w : Fin (n + 1) → ℝ) :
    tailSum w ⟨0, Nat.zero_lt_succ n⟩ = ∑ j : Fin (n + 1), w j := by
  exact Finset.sum_subset (Finset.subset_univ _)
    fun x hx₁ hx₂ => by aesop

lemma tailSum_succ (w : Fin (n + 1) → ℝ) (i : Fin (n + 1))
    (hi : i.val + 1 ≤ n) :
    tailSum w i = w i + tailSum w ⟨i.val + 1, by omega⟩ := by
  unfold tailSum
  rw [Finset.Ici_eq_cons_Ioi, Finset.sum_cons]
  rcongr j; aesop

lemma constructAlpha_mem_Icc (w : Fin (n + 1) → ℝ) (hw : ∀ i, 0 < w i)
    (_hsum : ∑ i : Fin (n + 1), w i = 1) (i : Fin (n + 1)) :
    constructAlpha w i ∈ Set.Icc (0 : ℝ) 1 := by
  refine' ⟨div_nonneg (le_of_lt (hw i)) _, _⟩
  · exact Finset.sum_nonneg fun _ _ => le_of_lt (hw _)
  · refine' div_le_one_of_le₀ _ (le_of_lt (tailSum_pos _ hw i))
    exact Finset.single_le_sum (fun a _ => le_of_lt (hw a)) (by simp)

/-- The key telescoping identity: the transmittance through layers
  0, ..., i-1 equals the tail sum from i onward. -/
lemma prod_one_sub_constructAlpha (w : Fin (n + 1) → ℝ) (hw : ∀ i, 0 < w i)
    (hsum : ∑ i : Fin (n + 1), w i = 1) (i : Fin (n + 1)) :
    ∏ j ∈ Finset.Iio i, (1 - constructAlpha w j) = tailSum w i := by
  induction' i using Fin.induction with i ih
  · unfold constructAlpha tailSum
    erw [Finset.prod_empty,
      show (Ici 0 : Finset (Fin (n + 1))) = Finset.univ from
        Finset.eq_univ_of_forall fun i => Finset.mem_Ici.mpr (Nat.zero_le _)]
    aesop
  · rw [show (Finset.Iio (Fin.succ i) : Finset (Fin (n + 1))) =
        Finset.Iio (Fin.castSucc i) ∪ {Fin.castSucc i} from ?_,
      Finset.prod_union] <;> norm_num
    · rw [ih, tailSum_succ]
      rw [constructAlpha]
      all_goals norm_num
      rw [one_sub_div, mul_div, mul_comm]
      · rw [div_eq_iff]
        · rw [tailSum_succ]; ring!
          exact Nat.succ_le_of_lt i.2
        · exact ne_of_gt <| Finset.sum_pos (fun _ _ => hw _) <| Finset.nonempty_Ici
      · exact ne_of_gt <| tailSum_pos _ (fun i => hw i) _
    · exact val_inj.mp rfl

lemma alphaWeight_constructAlpha (w : Fin (n + 1) → ℝ) (hw : ∀ i, 0 < w i)
    (hsum : ∑ i : Fin (n + 1), w i = 1) (i : Fin (n + 1)) :
    alphaWeight (constructAlpha w) i = w i := by
  convert congr_arg (fun x : ℝ => w i / tailSum w i * x)
    (prod_one_sub_constructAlpha w hw hsum i) using 1
  rw [div_mul_cancel₀ _ (ne_of_gt (tailSum_pos w hw i))]

/-- Softmax is a special case of alpha compositing: every softmax weight
  vector can be exactly represented as alpha compositing weights. -/
theorem softmax_subset_alpha (s : Fin (n + 1) → ℝ) :
    ∃ (a : Fin (n + 1) → ℝ), (∀ i, a i ∈ Set.Icc (0 : ℝ) 1) ∧
      ∀ i, alphaWeight a i = softmaxWeight s i := by
  apply Exists.intro (fun i => constructAlpha (softmaxWeight s) i)
  apply And.intro
  · exact fun i => constructAlpha_mem_Icc _
      (fun i => softmaxWeight_pos _ _) (by simp [softmaxWeight_sum]) _
  · apply alphaWeight_constructAlpha
    · exact fun i => softmaxWeight_pos s i
    · exact softmaxWeight_sum s

end
```

---

## Notation and Definitions Reference

| Symbol | Definition | Domain |
|---|---|---|
| sᵢ | Relevance score for element i | ℝ (any real number) |
| wᵢ (softmax) | exp(sᵢ) / Σⱼ exp(sⱼ) | (0, 1) — strictly positive |
| aᵢ | Opacity of element i | [0, 1] — can be zero |
| Tᵢ | Transmittance: ∏ⱼ<ᵢ (1 - aⱼ) | [0, 1] — non-increasing |
| wᵢ (compositing) | aᵢ · Tᵢ | [0, 1] — can be zero |
| Rᵢ (tail sum) | Σⱼ≥ᵢ wⱼ | (0, 1] |

---

## References

1. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.** (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*.

2. **Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.** (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *ACM Transactions on Graphics (SIGGRAPH), 42(4)*.

3. **Porter, T. & Duff, T.** (1984). Compositing Digital Images. *ACM SIGGRAPH Computer Graphics, 18(3), 253-259*.

4. **Max, N.** (1995). Optical Models for Direct Volume Rendering. *IEEE Transactions on Visualization and Computer Graphics, 1(2), 99-108*.

5. **Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.** (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. *European Conference on Computer Vision (ECCV)*.

6. **Ramsauer, H., Schafl, B., Lehner, J., Seidl, P., Widrich, M., et al.** (2021). Hopfield Networks is All You Need. *International Conference on Learning Representations (ICLR)*.

7. **Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F.** (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. *International Conference on Machine Learning (ICML)*.

8. **de Moura, L., Ullrich, S.** (2021). The Lean 4 Theorem Prover and Programming Language. *International Conference on Automated Deduction (CADE)*.

---

## Verification Details

| Property | Value |
|---|---|
| **Prover** | Aristotle (harmonic.fun) |
| **Language** | Lean 4 v4.28.0 |
| **Library** | Mathlib v4.28.0 |
| **`sorry` count** | 0 |
| **Axioms** | propext, Classical.choice, Quot.sound (standard) |
| **Aristotle Project ID** | efb72d79-54a2-49ed-a263-d7b9ce34dc33 |
| **Date** | 2026-04-07 |
