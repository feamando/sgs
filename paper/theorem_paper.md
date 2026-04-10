# On the Expressiveness of Alpha-Compositing: A Strict Superset of Softmax Attention

Nikita Gorshkov

---

## Abstract

We establish a formal relationship between two fundamental weighted aggregation schemes used in machine learning and computer graphics: softmax attention (the core operation of transformer architectures) and alpha-compositing (the rendering equation used in volume rendering and 3D Gaussian Splatting). We prove that the set of weight vectors achievable by alpha-compositing strictly contains those achievable by softmax attention — that is, every computation expressible through softmax can be exactly replicated by alpha-compositing, but not vice versa. Alpha-compositing additionally supports exact zero weights (hard sparsity) and sub-unity weight sums (residual capacity). Our proof is formally verified in Lean 4 with Mathlib, using only standard axioms and zero unproven assertions (`sorry`). We discuss implications for architecture design, connecting this result to the broader question of what inductive biases weighted aggregation schemes impose on learned representations.

**Keywords:** attention mechanisms, alpha-compositing, volume rendering, expressiveness, formal verification, Lean 4

---

## 1 Introduction

Weighted aggregation — computing an output as a weighted sum of input values — is arguably the most fundamental operation in modern machine learning. The specific scheme used to compute the weights determines the inductive bias of the architecture. Two prominent schemes have emerged independently in different fields:

**Softmax attention** (Vaswani et al., 2017), the core of the transformer architecture, computes weights as:
$$w_i^{\text{sfm}} = \frac{\exp(s_i)}{\sum_{j=1}^{n} \exp(s_j)}, \quad \text{output} = \sum_{i=1}^{n} w_i^{\text{sfm}} \cdot v_i$$

**Alpha-compositing** (Porter & Duff, 1984; Max, 1995), the standard rendering equation in computer graphics, computes weights as:
$$w_i^{\alpha} = a_i \prod_{j=1}^{i-1}(1 - a_j), \quad \text{output} = \sum_{i=1}^{n} w_i^{\alpha} \cdot v_i$$

Both produce a weighted sum of value vectors. Both are differentiable and amenable to gradient-based optimization. Despite their structural similarity, their mathematical properties differ in ways that have implications for architecture design. In this paper, we characterize the precise set-theoretic relationship between the two schemes and formally verify the result in the Lean 4 theorem prover.

### 1.1 Contributions

1. We prove that $\mathcal{W}_{\text{softmax}} \subsetneq \mathcal{W}_{\alpha}$: every softmax weight vector can be exactly produced by alpha-compositing, but not vice versa (Theorem 1).
2. We provide an explicit, constructive mapping from softmax weights to compositing opacities (Theorem 2).
3. We characterize the additional weight vectors alpha-compositing can produce: those with exact zeros and those with sub-unity sums (Corollaries 1-2).
4. The complete proof is formally verified in Lean 4 with Mathlib v4.28.0, compiled with zero `sorry` statements and only standard axioms.

---

## 2 Preliminaries

### 2.1 Notation

Let $n \geq 2$ be the number of elements. We write $\Delta^{n-1} = \{w \in \mathbb{R}^n : w_i \geq 0, \sum_i w_i = 1\}$ for the probability simplex, and $\Delta^{n-1}_{>0} = \{w \in \Delta^{n-1} : w_i > 0 \; \forall i\}$ for its interior (the open simplex).

### 2.2 Softmax Weights

Given scores $s = (s_1, \ldots, s_n) \in \mathbb{R}^n$, the softmax function produces:
$$\text{softmax}(s)_i = \frac{\exp(s_i)}{\sum_{j=1}^{n} \exp(s_j)}$$

**Properties.** For any $s \in \mathbb{R}^n$:
- (S1) Strict positivity: $\text{softmax}(s)_i > 0$ for all $i$, since $\exp(s_i) > 0$.
- (S2) Normalization: $\sum_i \text{softmax}(s)_i = 1$.
- (S3) Surjectivity onto the open simplex: $\text{softmax}: \mathbb{R}^n \to \Delta^{n-1}_{>0}$ is surjective. For any $w \in \Delta^{n-1}_{>0}$, setting $s_i = \ln(w_i)$ recovers $w$.

Thus the set of achievable softmax weight vectors is exactly $\mathcal{W}_{\text{softmax}} = \Delta^{n-1}_{>0}$: the open probability simplex.

### 2.3 Alpha-Compositing Weights

Given opacities $a = (a_1, \ldots, a_n) \in [0,1]^n$, alpha-compositing (Porter & Duff, 1984) produces weights via the "over" operator:
$$w_i^{\alpha}(a) = a_i \cdot T_i(a), \quad T_i(a) = \prod_{j=1}^{i-1}(1 - a_j)$$

where $T_i$ is the accumulated transmittance — the fraction of "capacity" not absorbed by preceding elements. By convention, $T_1 = 1$ (the empty product).

**Properties.** For any $a \in [0,1]^n$:
- (A1) Non-negativity: $w_i^{\alpha} \geq 0$ for all $i$.
- (A2) Can be zero: $w_i^{\alpha} = 0$ when $a_i = 0$ (element contributes nothing).
- (A3) Bounded sum: $\sum_i w_i^{\alpha} \leq 1$, with equality iff $T_{n+1} = 0$.
- (A4) Order-dependent: permuting $a$ changes the weights (unlike softmax, which is permutation-equivariant with respect to scores).

Property (A3) follows from the telescoping identity (Lemma 1 below). The set of achievable alpha-compositing weight vectors is:
$$\mathcal{W}_{\alpha} = \{w \in \mathbb{R}^n_{\geq 0} : \sum_i w_i \leq 1, \; \exists \text{ valid } a \in [0,1]^n \text{ producing } w\}$$

---

## 3 Main Results

### 3.1 Telescoping Lemma

**Lemma 1** (Telescoping). *For any $a \in [0,1]^n$:*
$$\sum_{i=1}^{n} w_i^{\alpha}(a) = 1 - T_{n+1}(a) = 1 - \prod_{i=1}^{n}(1 - a_i)$$

*Proof.* By induction on $n$. For $n = 1$: $w_1 = a_1 \cdot 1 = a_1 = 1 - (1 - a_1) = 1 - T_2$. For the inductive step, observe that $w_i = T_i - T_{i+1}$ since $T_{i+1} = T_i(1 - a_i)$ implies $a_i T_i = T_i - T_{i+1}$. Summing telescopes: $\sum_{i=1}^{n} w_i = T_1 - T_{n+1} = 1 - T_{n+1}$. $\square$

*Lean:* `alpha_compositing_sum`

### 3.2 Softmax Cannot Produce Zero Weights

**Theorem 1** (Non-inclusion). *$\mathcal{W}_{\alpha} \not\subseteq \mathcal{W}_{\text{softmax}}$.*

*Proof.* Consider $n = 2$ and $a = (0, 1)$. Then:
$$w_1^{\alpha} = 0 \cdot 1 = 0, \quad w_2^{\alpha} = 1 \cdot (1 - 0) = 1$$

So $w^{\alpha} = (0, 1) \in \mathcal{W}_{\alpha}$. But for any $s \in \mathbb{R}^2$, $\text{softmax}(s)_1 = \exp(s_1)/(\exp(s_1) + \exp(s_2)) > 0$. Since $(0, 1) \notin \Delta^{1}_{>0}$, we have $(0, 1) \notin \mathcal{W}_{\text{softmax}}$. $\square$

*Lean:* `alpha_not_subset_softmax`

### 3.3 Alpha-Compositing Can Reproduce Any Softmax

**Theorem 2** (Inclusion, constructive). *$\mathcal{W}_{\text{softmax}} \subseteq \mathcal{W}_{\alpha}$. Moreover, the mapping is constructive: given any $w \in \Delta^{n-1}_{>0}$, define*
$$a_i = \frac{w_i}{R_i}, \quad R_i = \sum_{j=i}^{n} w_j$$

*Then $a \in (0, 1]^n$ and $w^{\alpha}(a) = w$.*

*Proof.* We verify two things: (i) $a_i \in (0,1]$ for all $i$, and (ii) $w_i^{\alpha}(a) = w_i$ for all $i$.

**Part (i).** Since $w_i > 0$ and $R_i = \sum_{j \geq i} w_j > 0$, we have $a_i = w_i / R_i > 0$. Since $w_i \leq R_i$ (as $w_i$ is one term of the sum $R_i$), we have $a_i \leq 1$. Thus $a_i \in (0, 1]$.

**Part (ii).** We show by induction that $T_i(a) = R_i$ for all $i \in \{1, \ldots, n\}$.

*Base case* ($i = 1$): $T_1 = 1$ and $R_1 = \sum_{j=1}^{n} w_j = 1$ since $w \in \Delta^{n-1}$. ✓

*Inductive step:* Assume $T_i = R_i$. Then:
$$T_{i+1} = T_i \cdot (1 - a_i) = R_i \cdot \left(1 - \frac{w_i}{R_i}\right) = R_i - w_i = \sum_{j=i+1}^{n} w_j = R_{i+1}$$

This gives us $T_i = R_i$ for all $i$. Therefore:
$$w_i^{\alpha}(a) = a_i \cdot T_i = \frac{w_i}{R_i} \cdot R_i = w_i \quad \square$$

*Lean:* `softmax_subset_alpha`, with key lemma `prod_one_sub_constructAlpha`

### 3.4 Strict Inclusion

**Corollary 1** (Strict inclusion). *$\mathcal{W}_{\text{softmax}} \subsetneq \mathcal{W}_{\alpha}$.*

*Proof.* Immediate from Theorems 1 and 2. $\square$

**Corollary 2** (Characterization of the extra vectors). *$\mathcal{W}_{\alpha} \setminus \mathcal{W}_{\text{softmax}}$ contains:*
- *(a) All weight vectors with at least one zero entry and total sum 1 (the boundary of the simplex).*
- *(b) All non-negative weight vectors with $\sum_i w_i < 1$ (sub-simplex vectors).*

*Proof sketch.* (a) For any $w$ on the boundary of $\Delta^{n-1}$ with some $w_k = 0$: setting $a_k = 0$ and constructing remaining opacities via the Theorem 2 formula (restricted to non-zero entries) produces the desired weight vector. Since softmax cannot produce zero entries, these are in $\mathcal{W}_{\alpha} \setminus \mathcal{W}_{\text{softmax}}$. (b) For any target sum $S < 1$: choose opacities such that $T_{n+1} = 1 - S > 0$. By Lemma 1, the weights sum to $S$. Since softmax weights always sum to 1, these are excluded from $\mathcal{W}_{\text{softmax}}$. $\square$

---

## 4 Discussion

### 4.1 Implications for Architecture Design

Theorem 2 establishes that alpha-compositing can simulate softmax, but the converse fails. This has architectural implications:

**Hard sparsity.** Softmax attention always assigns positive weight to every element — there is no mechanism for completely ignoring irrelevant inputs. Alpha-compositing can set $a_i = 0$, producing exact zero contribution. In practice, this means alpha-compositing-based architectures can achieve hard attention without the non-differentiability issues of argmax-based selection.

**Residual capacity.** Softmax always distributes exactly 100% of weight across inputs. Alpha-compositing can distribute less than 100% ($\sum w_i < 1$), with the residual $1 - \sum w_i = T_{n+1}$ representing "unaccounted capacity." In a language model, this could represent uncertainty: the model has not fully committed its representation to the available inputs.

**Order sensitivity.** Alpha-compositing is inherently order-dependent (Property A4): changing the order of elements changes the weights. Softmax is order-agnostic (permutation-equivariant). This is a structural difference: alpha-compositing embeds sequential ordering into the weighting scheme, while softmax requires external positional encoding.

### 4.2 Expressiveness vs. Inductive Bias

We emphasize that Theorem 2 is a statement about the set of *achievable weight vectors*, not about which scheme produces better models. A more expressive scheme can represent more functions but also has a larger hypothesis space to search during optimization. The practical question is whether alpha-compositing's specific structure — transmittance-gated, order-dependent, supporting exact zeros — provides useful inductive bias for specific tasks.

Concurrent empirical work (Gorshkov, 2026) suggests that alpha-compositing provides a stronger inductive bias for sentence composition in NLP: better zero-shot and few-shot performance than softmax with limited data, converging to equivalent performance at scale. This is consistent with the theoretical picture: alpha-compositing's structural constraints (ordering, transmittance) encode useful compositional structure, while softmax's flexibility allows it to learn equivalent structure given sufficient data.

### 4.3 Connection to Prior Work

Ramsauer et al. (2021) proved that softmax attention is mathematically equivalent to the update rule of modern continuous Hopfield networks — framing attention as energy minimization. Katharopoulos et al. (2020) and Choromanski et al. (2021) showed that softmax attention can be decomposed as a kernel function, enabling linear-complexity approximations.

Our result adds a third characterization: softmax attention is a strict subset of alpha-compositing. Together with the Hopfield and kernel perspectives, this suggests a hierarchy of weighted aggregation schemes:

$$\text{Linear attention} \subseteq \text{Softmax attention} \subsetneq \text{Alpha-compositing}$$

Whether further strict inclusions exist (e.g., whether there are useful schemes strictly containing alpha-compositing) is an open question.

### 4.4 Connection to Computer Graphics

Alpha-compositing was introduced by Porter & Duff (1984) for digital image compositing and later formalized for volume rendering by Max (1995). The rendering equation used in 3D Gaussian Splatting (Kerbl et al., 2023) is a specific instantiation where the opacities $a_i$ are computed as the product of a learned base opacity $\alpha_i$ and a Gaussian kernel evaluation $\mathcal{K}(x, \mu_i, \Sigma_i)$.

Our theorem applies to the general alpha-compositing scheme, subsuming both the classical Porter-Duff operator and the 3DGS rendering equation as special cases. The result establishes a formal connection between the computational foundations of computer graphics and modern attention mechanisms.

---

## 5 Formal Verification

The complete proof is formalized in Lean 4 (de Moura & Ullrich, 2021) using the Mathlib library (v4.28.0). The formalization contains:

| Lean Definition/Theorem | Mathematical Object |
|---|---|
| `softmaxWeight s i` | $w_i^{\text{sfm}} = \exp(s_i) / \sum_j \exp(s_j)$ |
| `alphaWeight a i` | $w_i^{\alpha} = a_i \prod_{j<i}(1 - a_j)$ |
| `tailSum w i` | $R_i = \sum_{j \geq i} w_j$ |
| `constructAlpha w i` | $a_i = w_i / R_i$ |
| `softmaxWeight_pos` | $w_i^{\text{sfm}} > 0$ for all $i$ (Property S1) |
| `softmaxWeight_sum` | $\sum_i w_i^{\text{sfm}} = 1$ (Property S2) |
| `alpha_not_subset_softmax` | Theorem 1: $\mathcal{W}_{\alpha} \not\subseteq \mathcal{W}_{\text{sfm}}$ |
| `prod_one_sub_constructAlpha` | Key lemma: $T_i(a) = R_i$ (telescoping) |
| `softmax_subset_alpha` | Theorem 2: $\mathcal{W}_{\text{sfm}} \subseteq \mathcal{W}_{\alpha}$ |

The proof compiles with zero `sorry` statements and depends only on standard axioms: `propext`, `Classical.choice`, and `Quot.sound`. The Lean source is available at `https://github.com/feamando/sgs/docs/proofs/lean/claim_3_5_softmax_subset_alpha.lean`.

---

## 6 Conclusion

We have established that alpha-compositing is strictly more expressive than softmax attention as a weighted aggregation scheme. The constructive proof provides an explicit, efficiently computable mapping from any softmax weight vector to compositing opacities via the tail-sum formula $a_i = w_i / \sum_{j \geq i} w_j$. This result, verified in Lean 4, provides a theoretical foundation for exploring alpha-compositing as an alternative or complement to softmax attention in neural architectures.

---

## References

Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J., Mohiuddin, A., Kaiser, L., Belanger, D., Colwell, L., & Weller, A. (2021). Rethinking Attention with Performers. In *International Conference on Learning Representations*.

de Moura, L. & Ullrich, S. (2021). The Lean 4 Theorem Prover and Programming Language. In *International Conference on Automated Deduction*.

Gorshkov, N. (2026). Semantic Gaussian Splatting: Alpha-Compositing as a Composition Mechanism for Language. *Preprint*.

Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. In *International Conference on Machine Learning*.

Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *ACM Transactions on Graphics (SIGGRAPH)*, 42(4).

Max, N. (1995). Optical Models for Direct Volume Rendering. *IEEE Transactions on Visualization and Computer Graphics*, 1(2), 99-108.

Porter, T. & Duff, T. (1984). Compositing Digital Images. *ACM SIGGRAPH Computer Graphics*, 18(3), 253-259.

Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Adler, T., Gruber, L., Holzleitner, M., Pavlović, M., Sandve, G.K., Bock, C., Kreil, D., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2021). Hopfield Networks is All You Need. In *International Conference on Learning Representations*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. In *Advances in Neural Information Processing Systems*.
