# Orthogonal Challenge: Hierarchical SGS (Knowledge Splatting)

**Date:** 2026-04-14
**Target document:** `docs/whitepaper/hierarchical_sgs.md`
**Focus:** Mathematical soundness, risk analysis, architectural validity

---

## Methodology

Three-round challenge. Each round: identify the weakest claim, attack it maximally, then resolve.

---

## Round 1: The Transmittance Ordering Illusion

### Attack

The whitepaper claims transmittance-based composition is superior to standard attention because it creates ordering — earlier blobs consume capacity, suppressing redundant later blobs. Section 4.4 (Claim H4) then reveals that **total rendering weight is order-independent** (W_total = 1 - Π(1 - aᵢ), which is a commutative product).

This is a significant self-contradiction. The paper simultaneously claims:
1. Ordering is a key advantage of alpha-compositing over softmax
2. Total weight is independent of ordering

If the total weight is the same regardless of blob order, what exactly is the "transmittance advantage"? The only thing ordering changes is the DISTRIBUTION of weight across blobs — and the paper uses a heuristic (descending K·α) without proving it's optimal.

Worse: in standard multi-head attention, each head learns its own relevance weighting. This is strictly more flexible than a fixed ordering heuristic. The paper's "advantage" (transmittance suppresses redundancy) could equally be described as a "constraint" (first-mover advantage biases toward whatever blob ranks highest by K·α, regardless of whether it's the best for generation).

**Concrete failure mode:** Consider a query about "machine learning for climate prediction." Two blobs:
- Blob A: "Machine learning basics" (K=0.7, α=0.8, K·α=0.56)
- Blob B: "Climate prediction methods" (K=0.6, α=0.9, K·α=0.54)

Blob A renders first (higher K·α). But for this query, Blob B is arguably more useful (the user knows ML; they want climate info). The transmittance heuristic gets this wrong — the less informative blob dominates.

### Response

The attack correctly identifies that total weight is order-independent. But the distributional effect is exactly the point. Consider what happens with the two blobs:

- Order [A, B]: ML blob gets 56% weight, climate blob gets 54% × 44% ≈ 24% weight. Total: 80%.
- Order [B, A]: Climate blob gets 54% weight, ML blob gets 56% × 46% ≈ 26% weight. Total: 80%.

Same total, different distribution. The question is whether the K·α heuristic picks the right distribution. The paper acknowledges this is a heuristic and proposes learned ordering as an alternative.

However, the attack's concrete failure mode is valid. K·α ranks by "relevance to query × blob confidence" — but this doesn't account for "informativeness given what the model already knows." A query about "ML for climate" from an ML expert should prioritize the climate blob, but K·α doesn't know the user's background.

### Resolution

**Accepted as a real limitation.** The K·α heuristic optimizes for query-blob relevance, not for informativeness. Two mitigations:

1. **Learned ordering** (experiment ablation): train a small network that takes (query, blob_set) and outputs ordering. Compare against K·α heuristic. If learned ordering wins significantly, replace the heuristic.

2. **Note in the paper:** Ordering affects weight distribution, not total weight. The heuristic is a reasonable default but may be suboptimal for queries requiring information from low-K blobs. This is analogous to the "explore vs. exploit" problem — the heuristic exploits (most relevant first), but exploration (diversity) may sometimes be better.

**Mathematical action:** Add to Claim H4 that total weight invariance means ordering is a soft preference, not a hard constraint. The system is robust to ordering mistakes — the "wrong" blob rendering first doesn't destroy information, it just shifts the emphasis.

---

## Round 2: The Gaussian Cluster Assumption Is Dangerously Optimistic

### Attack

Section 4.3 acknowledges that blob construction via clustering assumes chunk meanings are approximately Gaussian within each cluster. The paper then says "validate empirically" — but this dodges the real problem.

Consider the actual training data:
- **TinyStories** contains ~2.1M stories. Clustering into 10K clusters gives ~210 stories per cluster. Many clusters will contain stories with the same THEME but different PLOTS. Example: a cluster about "animals in the park" would include stories where:
  - A dog chases a cat (action)
  - A bird sings on a bench (peaceful)
  - A child feeds ducks (nurturing)

These have similar TOPIC embeddings (animals, park) but very different NARRATIVE embeddings. The cluster centroid ("generic animal park story") is equidistant from all of them — and the Gaussian kernel K(q, μ_cluster, Σ_cluster) will respond identically to "dog chasing" and "bird singing" queries.

This means blob retrieval for TinyStories will return TOPICALLY correct but NARRATIVELY generic blobs. The blob says "animal park story" but doesn't help with "chase" vs. "sing" vs. "feed." The model will render a generic animal-park story, which is exactly the "too generic" failure the user worried about.

For FineWeb-Edu (Hertz 1.1), the problem is worse. Educational text about "photosynthesis" includes:
- Simple explanations for children
- Advanced biochemistry
- Lab protocols
- Historical discovery narrative

A single cluster for "photosynthesis" would have a centroid in the middle of all four — useless for any specific query.

### Response

The attack is correct that naive k-means clustering will produce over-broad clusters. But the whitepaper already proposes multi-resolution blobs (Section 5, Challenge Round 3): fine blobs (sentence-level, low Σ) capture specifics, coarse blobs (document-level, high Σ) capture topics.

For TinyStories: 10K clusters of 210 stories is too coarse. At 50K clusters (42 stories each), clusters would be much more specific — "stories where a dog chases something" vs. "stories about birds in parks." The ablation plan (n_blobs ∈ {1K, 5K, 10K, 50K}) covers this.

For FineWeb-Edu: the hierarchical scheme (10K coarse + 30K medium + 60K fine) should capture "photosynthesis" at the coarse level and "simple photosynthesis explanation" at the fine level. The kernel naturally activates the fine blob for a "explain photosynthesis simply" query.

### Resolution

**Partially accepted.** The criticism is valid for the 10K cluster plan — it's likely too coarse. Actions:

1. **Increase default n_blobs** for Planck 1.1 from 10K to 50K. The parameter cost is still manageable: 50K × 1257 ≈ 63M (63% of base 100M — aggressive but feasible).

2. **Add a cluster quality gate:** Before training Planck 1.1, measure intra-cluster coherence. If average cosine similarity within clusters is < 0.7, clusters are too broad — increase n_blobs or switch to finer clustering.

3. **Consider supervised clustering** for TinyStories: cluster by narrative pattern (rising action, falling action, moral) rather than by topic. This requires a small annotation effort but produces more meaningful blobs.

**Update to whitepaper:** Change Planck 1.1 default from 10K to a sweep starting at 50K. Add cluster quality metrics to the experiment protocol.

---

## Round 3: The ANN Approximation Hides a Covariance Problem

### Attack

Section 2.4 claims that Gaussian kernel with diagonal covariance reduces to weighted L2:

```
K(q, μ, Σ) = exp(-0.5 · Σᵢ (qᵢ - μᵢ)² / σᵢ²) = exp(-0.5 · ||D⁻¹(q - μ)||²)
```

And therefore FAISS can index blobs by applying D⁻¹ to all blob means and using standard L2. But this has a critical problem: **each blob has its own D = diag(σ_b)**. You can't pre-transform all blob means with a single D⁻¹ because each blob has a different covariance.

For standard L2 indexing, you need:
```
distance(q, μ_b) = ||q - μ_b||²
```

But the Gaussian kernel computes:
```
distance(q, μ_b) = Σᵢ (qᵢ - μ_{b,i})² / σ_{b,i}²
```

This is a PER-BLOB weighted distance — different blobs weight different dimensions differently. FAISS doesn't support this. You'd need to evaluate the full Gaussian kernel for every blob, which is O(n_blobs × d_s) per query — defeating the purpose of ANN indexing.

The paper's Section 4.7 acknowledges ANN approximation error but doesn't address this fundamental incompatibility between per-blob covariance and standard ANN indexes.

### Response

The attack is technically correct — per-blob diagonal covariance makes standard L2 indexing inexact. However:

1. **First-stage approximation:** Use L2 distance over μ_b (ignoring Σ) for ANN retrieval of top-100 candidates. This is fast and high-recall for nearby blobs.

2. **Second-stage reranking:** Evaluate the full Gaussian kernel K(q, μ_b, Σ_b) on the top-100 candidates. Select top-k from these. This is cheap (100 × d_s operations).

This two-stage approach is standard in ANN-augmented retrieval. The assumption is that the L2-nearest blobs have high overlap with the Gaussian-kernel-nearest blobs — which is true when Σ values are similar across blobs (the μ distance dominates). It breaks down when some blobs have very different Σ from others.

3. **Alternative:** If Σ_b values are clustered (e.g., fine blobs all have similar small Σ, coarse blobs all have similar large Σ), build a separate FAISS index per resolution level with a shared D⁻¹ per level.

### Resolution

**Accepted as a real engineering issue.** The whitepaper overstates the simplicity of ANN indexing with per-blob covariance. Actions:

1. **Clarify the two-stage retrieval protocol** in the architecture section: L2 over μ_b for candidate generation, full kernel evaluation for reranking.

2. **Add empirical measurement:** Report recall@k of L2-then-rerank vs. exact Gaussian kernel search. If recall < 90%, the approximation is too loose.

3. **Add per-resolution indexing** as the recommended approach for multi-resolution blob stores.

**Mathematical note:** The retrieval approximation error bound in Section 4.7 should be revised to account for the two-stage protocol. The bound should be: if L2 retrieval has recall@100 ≥ r₁, and reranking selects top-k, then the final recall@k ≥ r₁ × (k/100) in the worst case. For r₁ = 0.95 and k = 8, this gives recall@8 ≥ 7.6% — which is too low. The bound needs tightening, or the candidate pool needs to be larger (top-500).

---

## Summary of Changes Required

| Round | Finding | Severity | Action |
|-------|---------|----------|--------|
| 1 | Ordering heuristic may be suboptimal | Medium | Add learned ordering as ablation. Note total weight is order-independent. |
| 2 | 10K clusters too coarse for TinyStories | High | Increase to 50K default. Add cluster quality gate. |
| 3 | Per-blob covariance breaks standard ANN | High | Clarify two-stage retrieval. Add per-resolution indexing. Tighten error bounds. |

---

## Claims Validated

| Claim | Status |
|-------|--------|
| Two-pass rendering extends standard SGS | Unchallenged — mathematically sound |
| Transmittance handles blob redundancy | Validated with nuance: total weight is invariant, distribution changes |
| T_max prevents blob dominance | Unchallenged — straightforward clamping |
| Dynamic blob addition works | Unchallenged — follows from existing proofs |
| Softmax ⊂ Alpha-Compositing under T_max | Unchallenged — scaling preserves relative weights |

---

## Claims Weakened

| Claim | Original Strength | Post-Challenge |
|-------|-------------------|----------------|
| K·α ordering is optimal | Assumed | Heuristic only; learned ordering may be better |
| Clustering produces good blobs | Assumed | Depends heavily on n_blobs; 10K is too few |
| FAISS handles Gaussian kernel retrieval | Claimed | Requires two-stage protocol; per-blob Σ not directly indexable |

---

## New Proof Obligations Identified

| Proof | Description | Priority |
|-------|-------------|----------|
| H4-extended | Total rendering weight is invariant under permutation of blobs | High (straightforward) |
| H6 (new) | Two-stage retrieval recall bound given L2-first approximation | Medium (approximation theory) |
| H7 (new) | Under what conditions on Σ_b distribution does L2 ≈ Gaussian kernel ranking? | Medium (statistical, not Lean4) |
