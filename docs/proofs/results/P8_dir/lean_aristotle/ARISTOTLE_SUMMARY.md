# Summary of changes for run aed03b38-96aa-4156-a583-57e9c5229a7c
Successfully proved Claim P8 in `claim_P8_clone_split_similarity.lean`. All 7 theorems compile without `sorry` and use only standard axioms (`propext`, `Classical.choice`, `Quot.sound`).

**Theorems proved:**

1. **`exp_neg_lipschitz`** — `exp(-x)` is 1-Lipschitz on `[0, ∞)`: `|exp(-x) - exp(-y)| ≤ |x - y|` for `x, y ≥ 0`. Proved via the Mean Value Theorem.

2. **`rescaled_cos_lipschitz`** — The rescaled cosine similarity `(cos θ + 1)/2` is `(1/2)`-Lipschitz in the angle θ.

3. **`weighted_sum_lipschitz`** — A weighted sum of Lipschitz functions `∑ wₖ · fₖ` is Lipschitz with constant `∑ wₖ · Lₖ`. Proved via the triangle inequality for finite sums.

4. **`clone_split_similarity_bound`** (main theorem) — For material similarity `M(ε) = ∑ wₖ · fₖ(ε)` with weights summing to 1 and each `fₖ` being `Lₖ`-Lipschitz with `fₖ(0) = 1`, we have `M(ε) ≥ 1 - (∑ wₖ · Lₖ) · ε` for all `ε ≥ 0`.

5. **`material_similarity_self`** — Self-similarity: `M(0) = 1`.

6. **`exp_component_self`** — `exp(-|0|) = 1`, confirming self-similarity for exponential components.

7. **`exp_component_lipschitz`** — `exp(-|·|)` is 1-Lipschitz, confirming the Lipschitz hypothesis for exponential components.

The formalization abstracts the material similarity as a weighted combination of component functions (each Lipschitz in a perturbation parameter ε), capturing the key insight that small perturbations in position/scale from cloning or splitting produce small changes in M.