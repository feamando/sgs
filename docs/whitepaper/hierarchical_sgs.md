# Hierarchical Semantic Gaussian Splatting: Knowledge Rendering for Language Models

**Radiance Labs — Research Proposal**
**Authors:** Nikita Gorshkov
**Date:** 2026-04-14
**Status:** Draft — Pre-Experiment
**Depends on:** SGS core (paper/semantic_gaussian_splatting.md), Planck 1.0, Hertz 1.0

---

## Abstract

We propose **Hierarchical SGS** — an extension of Semantic Gaussian Splatting that introduces multi-scale knowledge representations into the rendering pipeline. Where standard SGS represents each token as a word-level Gaussian and composes sentence meaning via alpha-compositing, Hierarchical SGS adds a second level: **knowledge blobs** — Gaussian distributions that represent chunks of pre-computed knowledge (facts, patterns, templates) in the same splatting space as word Gaussians. At inference time, the model first renders relevant knowledge blobs (setting a semantic backdrop), then renders word-level Gaussians in the remaining transmittance capacity (adding specifics). This unifies retrieval and generation into a single rendering equation — unlike RAG systems, which bolt retrieval onto generation as a separate system, knowledge blobs are native Gaussians that participate in the same alpha-compositing pass. We call this **Knowledge Splatting**.

---

## 1. Motivation

### 1.1 The Retrieval Problem in Current LLMs

Transformer-based LLMs encode all knowledge in their weights. This creates three problems:

1. **Knowledge is opaque.** You cannot inspect which "piece of knowledge" produced a given output.
2. **Knowledge is frozen.** Updating a fact requires retraining or fine-tuning.
3. **Knowledge is redundant.** The same fact (e.g., "Berlin is the capital of Germany") is encoded across millions of parameters, not stored once and referenced.

RAG (Retrieval-Augmented Generation) addresses this by maintaining an external vector store. But RAG has its own problems:

- Retrieval and generation are **separate systems** with separate embedding spaces. The retriever doesn't know what the generator needs; the generator doesn't know what the retriever found ambiguous.
- Retrieved chunks are **flat vectors** (cosine similarity). A chunk about "European weather patterns" and a chunk about "Berlin April temperatures" have the same representational structure — there's no way to express that one is broad and uncertain while the other is narrow and specific.
- **No natural composition.** When multiple chunks are retrieved, they're concatenated or cross-attended independently. There's no mechanism for one chunk to "occlude" another (i.e., if chunk A fully answers the question, chunk B should contribute nothing).

### 1.2 The 3DGS Analogy

In 3D Gaussian Splatting, a scene contains Gaussians at multiple scales:

| Scale | Example | Properties |
|-------|---------|------------|
| Background | Sky, distant buildings | Large Σ, low α, broad coverage |
| Midground | Furniture, walls | Medium Σ, medium α |
| Foreground | Faces, text, detail | Small Σ, high α, precise |

The renderer doesn't treat these differently — they're all Gaussians, composited front-to-back. The hierarchy emerges from their properties: background Gaussians are large and translucent (high Σ, low α), foreground Gaussians are small and opaque (low Σ, high α). Transmittance naturally handles the interaction — background fills in where foreground doesn't cover.

**The same principle applies to language.** A response to "What's the weather in Berlin in April?" has:

| Scale | Content | Properties |
|-------|---------|------------|
| Knowledge backdrop | "Berlin spring: 5-15°C, rain common, warming through April" | Broad meaning, moderate confidence |
| Query-specific detail | "April 14", "today", user's actual question | Narrow meaning, high relevance |
| Word-level rendering | The actual tokens composing the response | Token-by-token generation |

Hierarchical SGS makes this explicit: knowledge blobs are the "background Gaussians" of language, and word-level tokens are the "foreground Gaussians."

### 1.3 What This Is Not

This is **not** standard RAG with a Gaussian distance metric. The key differences:

1. **Unified representation.** Blobs and words are both Gaussians in the same d_s space. Retrieval IS rendering — same kernel, same compositing equation.
2. **Transmittance-gated composition.** Blobs consume rendering capacity. If blob A fully answers the query, blob B gets suppressed — not by a separate re-ranking step, but by the physics of the rendering equation.
3. **Uncertainty is first-class.** A blob's Σ encodes semantic breadth. A narrow blob ("Berlin April high: 15°C") matches only specific queries. A broad blob ("European climate patterns") matches many queries weakly. Standard RAG can't express this.
4. **Built into the model.** Blobs participate in the forward pass and receive gradients during training. The model *learns* how to use blobs, rather than having retrieval bolted on post-hoc.

---

## 2. Architecture

### 2.1 Standard SGS Rendering (Recap)

The SGS rendering equation composes sentence meaning from word-level Gaussians:

```
M(q) = Σᵢ fᵢ · wᵢ

where:
  wᵢ = αᵢ · K(q, μᵢ, Σᵢ) · Tᵢ
  Tᵢ = Π_{j<i} (1 - αⱼ · K(q, μⱼ, Σⱼ))
```

- **q** — query point in splatting space (d_s dimensions)
- **μᵢ, Σᵢ** — mean and covariance of word Gaussian i
- **αᵢ** — learned opacity of word i
- **K** — Gaussian kernel (relevance of word i to query q)
- **Tᵢ** — transmittance (remaining rendering capacity at position i)
- **fᵢ** — feature vector of word i (d_f dimensions)

### 2.2 Hierarchical SGS: Two-Pass Rendering

We extend the rendering equation with a **blob pass** before the **word pass**:

```
Pass 1 (Knowledge Blobs):
  M_blob(q) = Σⱼ f_bⱼ · w_bⱼ

  where:
    w_bⱼ = α_bⱼ · K(q, μ_bⱼ, Σ_bⱼ) · T_bⱼ
    T_bⱼ = Π_{k<j} (1 - α_bk · K(q, μ_bk, Σ_bk))

Pass 2 (Word Tokens):
  M_word(q) = Σᵢ fᵢ · wᵢ · T_residual

  where:
    T_residual = Π_j (1 - α_bⱼ · K(q, μ_bⱼ, Σ_bⱼ))   [transmittance after all blobs]

Final meaning:
  M(q) = M_blob(q) + M_word(q)
```

**Interpretation:**
- Blobs render first, consuming transmittance proportional to their relevance × opacity.
- Words render in whatever capacity remains.
- If blobs fully cover the query (T_residual ≈ 0), word-level rendering contributes minimally — the output is dominated by retrieved knowledge.
- If no blob matches (all K_blob ≈ 0), T_residual ≈ 1 — pure word-level rendering, identical to standard SGS.
- Partial coverage: blobs set the semantic backdrop, words add specifics.

### 2.3 Blob Representation

Each knowledge blob is a Gaussian in the same splatting space as words:

```
Blob_j = {
  μ_bⱼ  ∈ ℝ^{d_s}    — centroid meaning (position in splatting space)
  Σ_bⱼ  ∈ ℝ^{d_s}    — semantic breadth (diagonal covariance)
  α_bⱼ  ∈ (0, 1)     — confidence / authority
  f_bⱼ  ∈ ℝ^{d_f}    — feature vector (semantic content)
  text_j: str         — source text (for traceability, not used in forward pass)
}
```

**Property semantics:**

| Property | Low value | High value |
|----------|-----------|------------|
| Σ_b (variance) | Specific fact ("Berlin April high: 15°C") | Broad topic ("European climate") |
| α_b (opacity) | Uncertain / noisy source | Verified / authoritative source |
| K(q, μ_b, Σ_b) | Query is far from blob's meaning | Query is directly about this blob |

### 2.4 Blob Retrieval

Evaluating K(q, μ_b, Σ_b) over all blobs is O(n_blobs × d_s). For large blob stores (>100K), we use approximate nearest-neighbor search:

1. **Pre-index** blob means μ_b using FAISS (IVF-PQ or HNSW).
2. At inference, retrieve **top-k** blobs by approximate Gaussian kernel distance.
3. Exact kernel evaluation only on the top-k candidates.
4. Alpha-composite the top-k blobs in the rendering pass.

With diagonal covariance, the Gaussian kernel reduces to weighted L2 distance, which standard ANN libraries handle efficiently:

```
K(q, μ, Σ) = exp(-0.5 · Σᵢ (qᵢ - μᵢ)² / σᵢ²)
            = exp(-0.5 · ||D⁻¹(q - μ)||²)   where D = diag(σ)
```

This is L2 distance in a transformed space — apply D⁻¹ to all blob means, index with standard L2, retrieve top-k.

### 2.5 Blob Store Construction

**Method 1: Clustering (Pre-training)**

Given a training corpus of N documents:

1. Split each document into chunks of ~128-512 tokens.
2. Encode each chunk using the SGS encoder: run the word-level rendering to get a meaning vector.
3. Cluster chunks by semantic similarity (k-means or hierarchical clustering).
4. For each cluster c:
   - μ_c = centroid of chunk meanings
   - Σ_c = variance of chunk meanings within cluster (encodes semantic breadth)
   - α_c = 1 / (1 + entropy of cluster) — tight clusters get high confidence
   - f_c = mean of chunk feature vectors (or trained via gradient descent)

**Method 2: Direct Encoding (Post-hoc / Dynamic)**

For adding knowledge at inference time without retraining:

1. Encode new text chunk using frozen SGS encoder.
2. Compute μ, Σ from the word Gaussians in the chunk.
3. Set α based on source confidence (e.g., verified = 0.9, user-provided = 0.5).
4. Compute f as the rendered meaning of the chunk.
5. Add to blob store index.

This is the **key advantage over pure parametric models**: new knowledge can be added by inserting a Gaussian into the blob store, with no weight updates.

**Method 3: Learned (During Training)**

Maintain a trainable blob embedding table (similar to a word embedding table but for chunks). During training:

1. For each training batch, compute input meaning.
2. Retrieve top-k blobs.
3. Render with two-pass equation.
4. Backpropagate loss through blob features f_b and blob opacity α_b.
5. Periodically re-index μ_b (since features change during training).

This is the most powerful approach but requires careful handling of the discrete top-k selection (straight-through estimator or Gumbel-softmax for gradient flow).

### 2.6 Integration with Autoregressive LM

In a causal language model (Planck/Hertz), the forward pass at each position t is:

```
Standard (Planck 1.0):
  x_1..t → SGS encoder → multi-pass rendering → logits_t → next token

Hierarchical (Planck 1.1):
  x_1..t → SGS encoder → input meaning q_t
  q_t → blob retrieval → top-k blobs
  → two-pass rendering (blobs + words) → logits_t → next token
```

**When does blob retrieval happen?**

- **Option A: Once per sequence.** Retrieve blobs based on the first few tokens (the "prompt"). Blobs condition the entire generation. Simple, fast, but can't adapt mid-generation.
- **Option B: Every N tokens.** Re-retrieve blobs every N steps. Allows the model to "look up" new knowledge as the generation evolves. More expensive.
- **Option C: Every token.** Maximum flexibility but maximum cost. Impractical for large blob stores.

**Recommended: Option A for initial experiments, Option B for production.**

### 2.7 Transmittance Budget

A critical design parameter: **T_blob_max** — the maximum transmittance that blobs can consume.

```
Clamp: effective blob opacity = min(α_bⱼ · K_bⱼ, T_blob_max / k)
```

Without this, an overconfident blob store could consume all transmittance, making the model a pure lookup system with no generative capacity. T_blob_max acts as a regularizer:

- T_blob_max = 0.0 → blobs disabled, pure word-level SGS
- T_blob_max = 0.5 → blobs can consume at most 50% of rendering capacity
- T_blob_max = 1.0 → blobs can fully dominate (not recommended)

**Start experiments at T_blob_max = 0.3** and sweep to find the optimal balance.

---

## 3. Core Parameters

| Parameter | Symbol | Description | Planck 1.1 | Hertz 1.1 |
|-----------|--------|-------------|------------|-----------|
| Blob count | n_blobs | Total blobs in store | 10,000 | 100,000 |
| Blob feature dim | d_b | Feature vector size (= d_f) | 1,000 | 3,700 |
| Splatting dim | d_s | Shared splatting space | 128 | 256 |
| Retrieve top-k | k | Blobs per query | 8 | 16 |
| Blob transmittance budget | T_max | Max blob contribution | 0.3 | 0.3 |
| Blob source chunk size | chunk_size | Tokens per source chunk | 128 | 256 |
| Retrieval frequency | retrieve_every | Re-retrieve interval | once/seq | 128 tokens |
| Blob learning rate | lr_blob | Separate LR for blob params | 1e-3 | 1e-4 |
| Blob construction | method | How blobs are created | Clustering | Clustering |

---

## 4. Mathematical Foundations and Proof Obligations

The Hierarchical SGS architecture makes several mathematical claims that require formal verification. Some follow from existing proofs (Claim 3.5: Softmax ⊂ Alpha-Compositing); others are new. We enumerate each claim, its status, and whether Lean 4 proofing is needed.

### 4.1 Claim H1: Two-Pass Equivalence (Partition Rendering)

**Statement:** Partitioning a sequence of Gaussians G = B ∪ W (blobs B, words W) and rendering B first then W produces identical output to rendering the concatenation [B; W] as a single sequence under the standard rendering equation.

```
Formally: Let G = (g_1, ..., g_k, g_{k+1}, ..., g_{k+n}) where g_1..g_k ∈ B and g_{k+1}..g_{k+n} ∈ W.

Then: M_single(q, G) = M_blob(q, B) + T_residual · M_word(q, W)

where T_residual = Π_{j=1}^{k} (1 - α_j · K_j)
```

**Status:** Should follow from the associativity of the transmittance product. The transmittance at position k+i in the single sequence decomposes as:

```
T_{k+i} = Π_{j<k+i} (1 - α_j · K_j)
         = [Π_{j<k} (1 - α_j · K_j)] · [Π_{k≤j<k+i} (1 - α_j · K_j)]
         = T_residual · T_i^{word}
```

This is a standard telescoping product identity.

**Lean 4 proof:** Required. Straightforward — extend `claim_3_1_weights_sum_bounded.lean` and `claim_3_2_monotonic_transmittance.lean` with a partition lemma. Estimated effort: 1 day.

**Risk if false:** Architecture is fundamentally flawed. (Very unlikely — this is basic algebra.)

### 4.2 Claim H2: Expressiveness Preservation Under T_max Cap

**Statement:** With a transmittance budget cap T_max ∈ (0, 1), the word-level rendering pass still satisfies Softmax ⊂ Alpha-Compositing within its available transmittance.

```
Formally: For any softmax weight vector w over n words with Σ wᵢ = 1, there exist
alpha-compositing parameters (α₁, ..., αₙ) such that the word-level weights under
initial transmittance T₀ = (1 - T_max) satisfy:

  w'ᵢ = T₀ · αᵢ · Π_{j<i}(1 - αⱼ) = T₀ · wᵢ

i.e., the relative weights are identical; only the absolute scale changes.
```

**Status:** The existing proof (Claim 3.5) constructs alpha values aᵢ = wᵢ / Σ_{j≥i} wⱼ to reproduce any softmax vector. This construction doesn't depend on the initial transmittance — the relative weights are preserved regardless of T₀. The proof should extend directly.

**Lean 4 proof:** Required. Extend `claim_3_5_softmax_subset_alpha.lean` with a scaling lemma. Estimated effort: 0.5 days.

**Risk if false:** The T_max cap could inadvertently make the word-level pass less expressive than softmax. (Unlikely — scaling preserves relative weights.)

### 4.3 Claim H3: Gaussian Sufficiency for Cluster Representations

**Statement:** If chunk meanings within a cluster are drawn from a distribution D, then representing the cluster as a single Gaussian N(μ_c, Σ_c) where μ_c = E[D] and Σ_c = Var[D] is the maximum-likelihood estimate under a Gaussian assumption.

**Status:** This is a standard result from statistics (MLE for normal distributions). However, the ASSUMPTION that chunk meanings are approximately Gaussian within each cluster is non-trivial. If a cluster is multi-modal (contains two sub-topics), the Gaussian centroid falls between them, and the kernel K(q, μ_c, Σ_c) will be inaccurate for both sub-topics.

**Lean 4 proof:** NOT required — this is a statistical assumption, not a mathematical theorem. It cannot be proven; it must be validated empirically.

**Empirical validation required:**
- After clustering, measure the kurtosis and skewness of each cluster.
- Clusters with high kurtosis (> 5) or bimodality should be split further.
- Report distribution of cluster quality metrics in the experiment.

**Risk if false:** Blob retrieval accuracy degrades for multi-modal clusters. Mitigation: use more clusters (finer granularity) or Gaussian Mixture Models per "blob" (but this increases complexity).

### 4.4 Claim H4: Blob Ordering by K·α Maximizes Information Contribution

**Statement:** Ordering blobs by descending K·α (effective opacity) before rendering maximizes the total information contributed by the blob pass, measured as the sum of effective weights.

```
Formally: For blobs (α₁·K₁, ..., αₖ·Kₖ), the sum of effective weights
  W_total = Σᵢ (αᵢ·Kᵢ) · Π_{j<i}(1 - αⱼ·Kⱼ)
is maximized when blobs are sorted in descending order of αᵢ·Kᵢ.
```

**Status:** This is **NOT obviously true** and may in fact be false.

Counter-example sketch: Consider two blobs with effective opacities 0.9 and 0.5.
- Order [0.9, 0.5]: W = 0.9 + 0.5 × 0.1 = 0.95
- Order [0.5, 0.9]: W = 0.5 + 0.9 × 0.5 = 0.95

In this case, total weight is identical (both sum to 0.95). For two elements, the total weight W = 1 - Π(1 - aᵢ) is independent of ordering — it's a product, hence commutative.

However, the DISTRIBUTION of weights across blobs changes with ordering:
- Order [0.9, 0.5]: blob 1 gets weight 0.9, blob 2 gets weight 0.05
- Order [0.5, 0.9]: blob 1 gets weight 0.5, blob 2 gets weight 0.45

The question is: which distribution is better for generation? Descending order concentrates weight on the most relevant blob (greedy). Ascending order distributes weight more evenly.

**Lean 4 proof:** Required — but to prove what? The total weight claim is false (ordering doesn't affect total weight). What we actually want is: does ordering by K·α produce the best GENERATION quality? This is an empirical question, not a mathematical one.

**Revised claim:** The total rendering weight Σ wᵢ = 1 - Π(1 - αᵢ·Kᵢ) is ORDER-INDEPENDENT. Ordering only affects the distribution of weight across blobs. Descending order by K·α is a heuristic that concentrates weight on the most relevant blob.

**Lean 4 proof required:** Prove that W_total = 1 - Π(1 - aᵢ) is permutation-invariant. Estimated effort: 0.5 days. (This follows from commutativity of multiplication.)

**Experimental validation required:** Compare descending K·α ordering vs. random ordering vs. learned ordering. If ordering doesn't matter (because W_total is invariant), the simpler random ordering suffices.

### 4.5 Claim H5: Dynamic Blob Addition Preserves Rendering Properties

**Statement:** Adding a new blob to an existing blob store at inference time does not violate any rendering equation guarantees (weight boundedness, transmittance monotonicity, etc.).

**Status:** All existing proofs (Claims 3.1, 3.2, 5.1) are parameterized by n (number of elements). Adding a blob increases n by 1. The proofs hold for arbitrary n, so they hold for n+1.

**Lean 4 proof:** NOT required — covered by existing proofs which are universally quantified over n.

**Risk if false:** None — this is a direct consequence of the existing proof structure.

### 4.6 Summary of Proof Obligations

| Claim | Statement | Lean 4? | Effort | Priority |
|-------|-----------|---------|--------|----------|
| H1 | Two-pass partition equivalence | Yes | 1 day | Critical |
| H2 | Expressiveness under T_max cap | Yes | 0.5 day | Critical |
| H3 | Gaussian sufficiency for clusters | No (empirical) | N/A | Medium |
| H4 | W_total is order-independent | Yes | 0.5 day | High |
| H4b | Descending K·α is optimal ordering | No (empirical) | N/A | Medium |
| H5 | Dynamic addition preserves properties | No (covered) | N/A | Low |

**Total new Lean 4 work: ~2 days, 3 new proof files.**

### 4.7 Assumption: ANN Approximation Error Bound

The architecture uses approximate nearest-neighbor search for blob retrieval. This introduces an error: the true top-k blobs may differ from the retrieved top-k.

**Bound needed:** If ANN retrieval has recall@k ≥ r (i.e., at least r fraction of true top-k are retrieved), then the rendering error is bounded by:

```
||M_approx - M_exact|| ≤ (1 - r) · T_max · max_j ||f_bⱼ||
```

This bounds the worst-case impact of approximate retrieval. For r ≥ 0.95 (standard for HNSW) and T_max = 0.3, the error is ≤ 0.015 × max blob feature norm.

**Lean 4 proof:** Desirable but not critical — standard approximation theory. Could be done as a stretch goal.

---

## 5. Orthogonal Challenge (Round 1 — Pre-Experiment)

### Challenge Round 1: "This is just attention over a key-value memory"

**Argument:** The blob rendering pass is mathematically equivalent to a cross-attention layer where keys=μ_b, values=f_b, and the attention kernel is Gaussian instead of dot-product. Calling it "rendering" is marketing, not substance. RETRO (Borgeaud et al., 2022) already does this with standard attention.

**Response:** Three structural differences separate this from cross-attention over a KV store:

1. **Transmittance creates ordering.** In attention, all keys contribute independently — their weights sum to 1 (softmax normalization). In rendering, earlier blobs consume capacity, reducing later blobs' contribution. This means the ORDER of blobs matters, and naturally handles redundancy: if two blobs say the same thing, the second one gets suppressed. Cross-attention doesn't do this.

2. **Variance is first-class.** Each blob has its own Σ, encoding how specific or broad it is. A narrow blob only activates for very specific queries; a broad blob activates for many queries weakly. In standard cross-attention, keys are points (no uncertainty). You'd need to add a learned temperature per key to approximate this — which is exactly what Σ provides.

3. **Shared space with generation.** Blobs and words are in the same splatting space. The model doesn't need to learn a separate "retrieval embedding" and "generation embedding" — they're unified. This means a blob can directly interact with word Gaussians through the rendering equation.

**Counter-argument:** These are incremental differences, not fundamental. The ordering from transmittance could be a constraint that hurts performance — what if the optimal composition is non-sequential? And shared space could be limiting: retrieval may benefit from a different metric than generation.

**Resolution:** Empirically testable. Compare:
- Hierarchical SGS (our proposal)
- Standard cross-attention over the same blob features
- RETRO-style chunked cross-attention

If transmittance ordering and shared space help, Hierarchical SGS wins. If not, we learn that the retrieval and generation spaces should be separate. **Either outcome is valuable.**

### Challenge Round 2: "Blobs will make the model lazy"

**Argument:** If the model can retrieve a pre-computed answer, it will learn to depend on retrieval instead of developing strong generative capabilities. The model becomes a blob lookup table that degrades catastrophically for any query outside blob coverage.

**Response:** Three mitigations:

1. **T_blob_max caps blob influence.** At T_max = 0.3, blobs can contribute at most 30% of the output. The remaining 70% must come from word-level rendering. The model cannot become a pure lookup system.

2. **Training signal forces generation.** The loss function is cross-entropy on next token prediction. If blobs hurt generation (e.g., by injecting irrelevant content), gradient will reduce blob α. The model learns when blobs help and when they don't.

3. **OOD evaluation is mandatory.** We explicitly test on out-of-distribution queries (novel story themes for Planck, knowledge questions about topics not in the blob store for Hertz). If OOD performance degrades relative to the base model, T_max is too high.

**Counter-argument:** Even at T_max = 0.3, the model may learn to "lean on" blobs for the easy 30% and become worse at the remaining 70% — because training is optimized for the combined output, not the word-level output alone.

**Resolution:** Ablation at test time: evaluate Planck 1.1 with blobs disabled (T_max = 0) and compare to Planck 1.0 base. If word-level generation degrades, blobs are harming the model. This is a **hard gate for the experiment — if base generation degrades, the approach fails.**

### Challenge Round 3: "Blob granularity is unsolvable"

**Argument:** There is no correct granularity for blobs. Too fine (sentence-level): you need millions of blobs, slow retrieval, high memory. Too coarse (paragraph-level): blobs are too generic, responses are vague. The optimal granularity depends on the query, which you don't know in advance.

**Response:** Multi-resolution blobs. Create blobs at multiple granularities:

- **Fine blobs** (sentence-level, low Σ): specific facts, match narrow queries
- **Medium blobs** (paragraph-level, medium Σ): topic summaries
- **Coarse blobs** (document-level, high Σ): broad context

The Gaussian kernel naturally handles multi-resolution retrieval. A specific query activates fine blobs (low Σ → sharp kernel peak). A vague query activates coarse blobs (high Σ → broad kernel response). No separate granularity decision needed — Σ encodes it.

**Counter-argument:** This multiplies the blob store size by 3× and makes training harder. Also, how do you prevent medium blobs from being redundant with fine blobs? They'll contain overlapping information.

**Resolution:** Transmittance handles redundancy — if a fine blob already rendered the specific fact, the medium blob that also contains it gets suppressed. This is literally what transmittance is for. Memory cost is real; budget accordingly (3× n_blobs).

### Challenge Round 4: "This won't work for code (Idea 2)"

**Argument:** Code has strict syntax. A "blob" of boilerplate code is useless unless it's syntactically complete and compatible with the surrounding code. Alpha-compositing doesn't respect syntax — it operates in semantic space, not token space.

**Response:** The blob's feature vector f_b conditions the next-token prediction, it doesn't directly produce output tokens. The autoregressive decoder still generates tokens one at a time, maintaining syntactic validity. The blob shifts the probability distribution toward tokens that are consistent with the template — like how a prompt steers generation without producing tokens itself.

**Counter-argument:** If the blob only "steers" the distribution, its influence may be too weak to produce the structured output needed for code. A dashboard template requires exact import statements, specific function signatures, precise CSS properties. Soft semantic steering may not be enough.

**Resolution:** This challenge is valid and has led to extracting code generation into a separate track (Track E: Radiance Zuse) with its own architecture where Gaussian variance represents implementation variability rather than semantic breadth. See `docs/whitepaper/sgs_code.md` for the dedicated approach. **H-SGS Knowledge Splatting focuses on natural language only.**

### Challenge Round 5: "The training data quality ceiling"

**Argument (the user's own concern):** Blobs are extracted from training data. The model can never produce output better than its blob store. For creative or novel tasks, blobs are a liability — they anchor the output to the generic patterns from training. This is the fundamental tension between retrieval (accurate but generic) and generation (creative but unreliable).

**Response:** This tension is real and cannot be engineered away. But it can be managed:

1. **The model already has this limitation.** All model knowledge comes from training data — whether stored in weights or in blobs. Blobs make this explicit and inspectable, not worse.

2. **T_blob_max is the dial.** High T_max → more retrieval influence (accurate, generic). Low T_max → more generation influence (creative, unreliable). Different applications need different settings.

3. **Dynamic blobs break the ceiling.** New knowledge can be added at inference time. Unlike frozen weights, the blob store can grow. A user can add their own code templates, domain knowledge, or current facts.

**Resolution:** Accepted as a fundamental trade-off, not a flaw. The experiment must measure both accuracy (where blobs help) and creativity (where blobs may hinder). Use perplexity for accuracy, human evaluation for creativity.

---

## 6. Experiment Plan

### 6.1 Planck 1.1 — TinyStories with Knowledge Blobs

**Objective:** Validate the Hierarchical SGS architecture on a small, controlled task. Does blob conditioning improve story coherence and reduce repetition?

**Base:** Planck 1.0 (100.9M params, d_s=128, d_f=1000, trained on TinyStories)

**Blob construction:**
1. Take the TinyStories training set (~2.1M stories).
2. Cluster stories by semantic similarity into 10,000 clusters.
3. For each cluster: extract μ_b (centroid), Σ_b (spread), f_b (mean feature).
4. The blob store represents 10K "story archetypes" — common narrative patterns.

**Model modification:**
- Add a BlobStore module: n_blobs=10,000, d_s=128, d_f=1000.
- Add two-pass rendering to the forward pass.
- Total new parameters: 10K × (128 + 128 + 1 + 1000) ≈ 12.6M (12% increase over 100.9M base).
- Freeze word-level parameters initially; train only blob parameters for 1 epoch. Then unfreeze and jointly train for 2 more epochs.

**Evaluation:**

| Metric | Method | What it tests |
|--------|--------|---------------|
| Perplexity | Validation set | Does blob conditioning reduce loss? |
| Repetition rate | % of 4-grams repeated in 200-token generations | Do blobs reduce repetition loops? |
| Story coherence | Human evaluation (20 stories, blind comparison) | Do blobs improve narrative structure? |
| Base generation quality | T_max=0 at eval time | Did blobs hurt word-level generation? |
| Blob utilization | Mean effective blob weight across generations | Are blobs actually being used? |

**Ablations:**
- n_blobs ∈ {1K, 5K, 10K, 50K}
- k ∈ {4, 8, 16}
- T_max ∈ {0.1, 0.2, 0.3, 0.5}
- Blob construction: clustering vs. random sampling vs. learned

**Hard gates (experiment fails if):**
- Base generation quality (T_max=0) worse than Planck 1.0
- Blob utilization < 5% (blobs are ignored)
- Perplexity doesn't improve over Planck 1.0

**Timeline:** After Planck 1.0 training is confirmed complete. ~1 week for blob construction + training.

### 6.2 Hertz 1.1 — FineWeb-Edu with Knowledge Blobs

**Objective:** Test at 1B scale with diverse knowledge. Can blobs improve factual accuracy? Can new knowledge be added at inference time?

**Base:** Hertz 1.0 (1.03B params, d_s=256, d_f=3700, trained on FineWeb-Edu)

**Blob construction:**
1. Chunk FineWeb-Edu into ~256-token passages.
2. Encode using frozen Hertz 1.0 encoder.
3. Cluster into 100K blobs (hierarchical: 10K coarse + 30K medium + 60K fine).
4. Multi-resolution blob store.

**Model modification:**
- BlobStore: n_blobs=100K, d_s=256, d_f=3700 (multi-resolution).
- New parameters: 100K × (256 + 256 + 1 + 3700) ≈ 421M (41% increase — significant).
- Consider d_b < d_f for blobs to reduce memory: d_b=1024 with a projection layer.
- With d_b=1024: 100K × (256 + 256 + 1 + 1024) ≈ 154M (15% increase — acceptable).

**Evaluation:**

| Metric | Method | What it tests |
|--------|--------|---------------|
| Perplexity | Validation set | General quality improvement |
| Knowledge QA | TriviaQA / Natural Questions subset | Factual accuracy from blobs |
| Dynamic knowledge | Add new facts as blobs, test retrieval | Can the model use new knowledge without retraining? |
| OOD generation | Generate on topics absent from blob store | Does OOD quality degrade? |
| Attribution | Track which blobs contributed to each output | Can we trace claims to sources? |

**Key experiment: Dynamic knowledge addition**
1. Train Hertz 1.1 on FineWeb-Edu blobs.
2. At test time, add 100 new facts as blobs (e.g., "The 2026 Olympics are in Milan-Cortina").
3. Query the model about these facts.
4. Measure: does the model correctly retrieve and render the new facts?

If this works, it demonstrates the core advantage over parametric models: **knowledge updates without retraining.**

**Hard gates:**
- OOD generation quality worse than Hertz 1.0
- Knowledge QA accuracy doesn't improve over Hertz 1.0
- Dynamic knowledge retrieval precision < 50%

**Timeline:** After Hertz 1.0 training completes. ~2 weeks for blob construction + training + evaluation.

### 6.3 Code: Extracted to Separate Track

Code generation requires fundamentally different Gaussian semantics (see `docs/whitepaper/sgs_code.md`). Extracted as **Track E: Radiance Zuse** — a dedicated code generation model line with its own architecture and blob semantics.

---

## 7. Comparison to Existing Work

| System | Knowledge Store | Retrieval Metric | Composition | Unified? |
|--------|----------------|-------------------|-------------|----------|
| Standard LLM | Weights (implicit) | N/A | Attention layers | Yes |
| RAG | Vector DB (flat) | Cosine similarity | Concatenation / cross-attention | No |
| RETRO | Chunk DB (flat) | BERT similarity | Chunked cross-attention | No |
| kNN-LM | Token-level DB | L2 distance | Interpolation | Partially |
| **Hierarchical SGS** | **Gaussian blob store** | **Gaussian kernel** | **Alpha-compositing** | **Yes** |

**Unique properties of Hierarchical SGS:**
1. Retrieval and generation use the same kernel and compositing equation.
2. Blobs encode uncertainty (Σ) and confidence (α) — not just position.
3. Transmittance handles multi-blob redundancy without re-ranking.
4. Dynamic blob addition uses the same representation as training-time blobs.

---

## 8. Risks and Mitigations

| Risk | Severity | Mitigation | Detection |
|------|----------|------------|-----------|
| Blob dominance (model becomes lookup) | High | T_max cap, mandatory OOD eval | Base generation quality at T_max=0 |
| Training instability (discrete top-k) | Medium | Straight-through estimator, warm start | Loss divergence monitoring |
| Memory scaling (large blob stores) | Medium | Reduced d_b with projection, ANN indexing | Memory profiling |
| Variance collapse (all Σ → 0 or ∞) | Medium | Σ regularization, init from cluster spread | Monitor Σ distribution during training |
| Blob staleness (frozen knowledge) | Low | Dynamic blob addition protocol | Compare static vs. dynamic blob eval |
| Code generation fails (syntax incompatibility) | N/A | Extracted to Track E: Radiance Zuse | See `docs/whitepaper/sgs_code.md` |

---

## 9. Open Questions

1. **Should blob ordering be learned or fixed by K·α?** Fixed ordering is simpler but may be suboptimal. Learned ordering adds parameters but could improve composition.

2. **Can blobs be shared across models?** If Planck 1.1 and Hertz 1.1 use different d_s, their blob stores are incompatible. A universal splatting space would enable blob sharing — but this may conflict with per-model optimization.

3. **What is the optimal blob lifecycle?** Should blobs be periodically re-clustered as the model trains? Or fixed at initialization? Re-clustering is expensive but prevents drift.

4. **Can this replace fine-tuning?** Instead of fine-tuning on domain data, could you add domain-specific blobs? This would be a much cheaper form of adaptation. Worth testing.

5. **Does this help with hallucination?** If the model preferentially renders from attributed blobs rather than generating from weights, hallucination should decrease. But if blob coverage is sparse, the model may hallucinate more in gaps. Net effect unclear.

---

## 10. Naming

Formal: **Hierarchical Semantic Gaussian Splatting (H-SGS)**
Colloquial: **Knowledge Splatting**
3DGS analogy: Multi-scale rendering (LOD)
Model versions: Planck 1.1, Hertz 1.1 (same base architecture + blob store)
Track designation: **Track B2** (extension of B1 language models)
Code generation: Extracted to **Track E: Radiance Zuse** (see `docs/whitepaper/sgs_code.md`)

---

## 11. References

- Lewis, P. et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS.
- Borgeaud, S. et al. (2022). "Improving Language Models by Retrieving from Trillions of Tokens." ICML. (RETRO)
- Khandelwal, U. et al. (2020). "Generalization through Memorization: Nearest Neighbor Language Models." ICLR. (kNN-LM)
- Izacard, G. et al. (2022). "Atlas: Few-shot Learning with Retrieval Augmented Language Models."
- Kerbl, B. et al. (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering." SIGGRAPH.
- Gorshkov, N. (2026). "Semantic Gaussian Splatting: Alpha-Compositing as a Composition Mechanism for Language."
