# Article 6: SGS Architecture — The Full Machine

*From theory to blueprint: every component of the system*

---

## Overview

The previous articles established the concepts. This one assembles them into a working architecture. SGS has seven atomic components — think of them as the parts of an engine. Each has a specific role, and the system only works when they're connected.

```
Input tokens
    ↓
[A1] Activate Semantic Gaussians → Gaussian Scene
    ↓
[A5] Multi-Pass Iterative Rendering:
    ┌──────────────────────────────────┐
    │ For each pass p = 1..P:          │
    │   [A4] Project via Viewpoint     │
    │   [A2] Evaluate Gaussian Kernel  │
    │   [A3] Alpha-Composite           │
    │   [A6] Apply Operators           │
    │   Update Gaussian parameters     │
    │   FFN on features                │
    └──────────────────────────────────┘
    ↓
Rendered Meaning Vector
    ↓
Decode → Output Token
    ↓
[A7] Adaptive Density (during training)
```

---

## The Two Spaces

SGS operates in two spaces simultaneously — this is a key architectural decision.

**Splatting Space (d_s = 64 dimensions):** This is where Gaussians live and where the rendering equation operates. It's low-dimensional because Gaussian evaluation breaks down in high dimensions (a Gaussian in 768 dimensions evaluates to numerically zero everywhere). Think of this as the "spatial layout" of meanings — where words are positioned relative to each other.

**Feature Space (d_f = 512 dimensions):** This is what each Gaussian *carries*. Rich semantic content — sentiment, part of speech, argument structure, domain markers. Think of this as the "payload" — the information that gets composited into the output.

The analogy to 3DGS: splatting space is like 3D (x,y,z) coordinates — where blobs are positioned. Feature space is like the spherical harmonic coefficients — what the blobs look like (color, shininess).

The rendering equation uses splatting space to compute *weights* (how much does each Gaussian contribute?) and feature space for *values* (what does it contribute?):

```
Meaning(q) = Σᵢ fᵢ · αᵢ · K(q, μᵢ, Σᵢ) · Tᵢ
              ↑           ↑                 ↑
           features    kernel in         transmittance
           (d_f=512)   splatting space   from splatting
                       (d_s=64)          space
```

---

## Component 1: The Semantic Gaussian (A1)

Each word in the vocabulary is stored as a Gaussian:

```
G = (μ, L, α, f)
```

| Part | Dimensions | What It Encodes |
|---|---|---|
| **μ** (mean) | 64 numbers | Position in splatting space — the word's central meaning |
| **L** (Cholesky factor) | 2,080 numbers | Shape/spread of the Gaussian — semantic breadth, directional uncertainty |
| **α** (opacity) | 1 number | Base salience — how "loud" this word is by default |
| **f** (features) | 512 numbers | Rich semantic payload — the content that gets composited |

Total: **2,657 parameters per word.**

For a 50K vocabulary: 50,000 × 2,657 ≈ **133M parameters** in the vocabulary alone. This is comparable to BERT-base's 110M parameters — the vocabulary IS the model, in a sense.

**Polysemy:** Words with multiple meanings get multiple Gaussians. "Bank" might have three: one centered in the finance region, one in the geography region, one in the verb region. Which one(s) activate depends on context (resolved in multi-pass rendering).

**Initialization:** Start from pre-trained GloVe embeddings (projected to d_s=64 via PCA for means, original vectors as features). Initialize covariances from word frequency (rare words → broader Gaussians) and polysemy counts.

---

## Component 2: The Gaussian Kernel (A2)

When a query point q asks "what meaning exists here?", the kernel evaluates how much each Gaussian contributes:

```
K(q, μ, Σ) = exp(-½ · (q-μ)^T Σ^{-1} (q-μ) / τ)
```

The temperature τ is critical. Without it, in 64 dimensions, the kernel would evaluate to essentially zero for any point that isn't exactly at the Gaussian's center. With τ = d_s = 64, the expected kernel value for a point drawn from the Gaussian itself is exp(-0.5) ≈ 0.61 — comfortably non-zero.

**Sparsity:** Points far from a Gaussian (beyond ~4 standard deviations) get kernel values below 10^{-3} — effectively zero. This means only a handful of Gaussians contribute to any given query, not the entire vocabulary. This is the source of SGS's potential efficiency advantage over attention's O(n²) computation.

---

## Component 3: The Rendering Equation (A3)

Given the activated Gaussian scene and a query:

```
Meaning(q) = Σᵢ fᵢ · wᵢ

where wᵢ = αᵢ · K(q, μᵢ, Σᵢ) · Tᵢ
      Tᵢ = ∏ⱼ<ᵢ (1 - αⱼ · K(q, μⱼ, Σⱼ))
```

Gaussians are composited in **sequence order** (left to right in the sentence). This means earlier words have more transmittance available — they get "first dibs" on contributing to the meaning.

This front-loading is corrected in multi-pass rendering (Component 5): later passes can increase opacity for later words, counteracting the positional bias.

---

## Component 4: Semantic Viewpoints (A4)

A viewpoint is a projection from the full splatting space into a task-specific subspace:

```
Viewpoint = (P, q)
  P: 64 → m projection matrix (reduces dimensions)
  q: query point in the m-dimensional projected space
```

Gaussians are projected: their means and covariances are transformed by P. Then the rendering equation operates in the projected space.

**Multi-view rendering** (analogous to multi-head attention): run H viewpoints in parallel, each with its own learned P and q. Concatenate results.

```
For head h = 1..H:
  Project all Gaussians via P_h
  Render from q_h
  Get Meaning_h (d_f-dimensional vector)

Output = Concat(Meaning_1, ..., Meaning_H) × W_out
```

With H=8 heads and d_f=512, each head produces a 64-dimensional output, concatenated to 512 and projected via W_out.

Different heads learn to extract different semantic aspects: one might specialize in agent/subject extraction, another in sentiment, another in temporal relations. This mirrors the head specialization observed in transformer attention.

---

## Component 5: Multi-Pass Rendering (A5)

A single rendering pass is like a single transformer layer — not enough for complex understanding. SGS uses P=8 passes, each refining the Gaussian scene:

```
For pass p = 1, 2, ..., 8:

  1. RENDER: compute multi-view rendering from current Gaussian parameters
  
  2. BROADCAST: for each Gaussian, render the scene AT ITS OWN POSITION
     → each Gaussian now "knows" what its neighborhood looks like
  
  3. UPDATE POSITIONS: shift Gaussian means based on context
     μ_i ← μ_i + tanh(MLP(f_i, context_i))
     → "bank" shifts toward finance or river based on context
  
  4. UPDATE SALIENCE: gate opacity based on context
     α_i ← α_i × sigmoid(MLP(f_i, context_i))
     → wrong sense of "bank" fades out (opacity → 0)
     → right sense stays strong
  
  5. UPDATE FEATURES: feedforward transformation
     f_i ← f_i + FFN(f_i, context_i)
     → enrich features with contextual information
```

**What each pass does:**
- **Pass 1:** Lexical — Gaussians sit at their default positions. Basic proximity-based composition.
- **Pass 2:** Disambiguation — "bank" near "river" shifts toward the river sense. "Bank" near "account" shifts toward finance.
- **Pass 3:** Syntactic integration — subject/object roles become encoded through positional relationships.
- **Pass 4-8:** Deep semantics — co-reference, pragmatics, long-range dependencies.

**The key insight:** In transformers, each layer recomputes attention from scratch over the same tokens. In SGS, each pass *moves the Gaussians* — the scene itself changes between passes. Disambiguation is not re-weighting the same elements; it's physically reshaping the semantic landscape.

---

## Component 6: Operator Gaussians (A6)

Standard alpha-blending can only add. But language needs:
- **Negation:** "not happy" — meaning should be anti-happy
- **Quantification:** "every student" vs. "some students" — different logical scope
- **Subordination:** "the dog that the cat chased" — nested structure

Operator Gaussians are special Gaussians that modify the rendering process instead of contributing content:

```
Each Gaussian has a soft type:
  type_probs = softmax(W_type · f_i)
  → [p_content, p_negate, p_quantifier, ...]
```

Most words are ~100% CONTENT. Function words like "not" learn high p_negate. Quantifiers like "every" learn high p_quantifier. This is learned, not hardcoded.

When a NEGATE-type Gaussian is encountered, it flips the sign of the next content Gaussian's feature contribution. This is the weakest component of SGS — it's a pragmatic solution, not a principled one. The expectation is that most non-monotonic composition (negation, scope) will be learned by the FFN in multi-pass rendering rather than by the operator mechanism.

---

## Component 7: Autoregressive Generation

To generate text, SGS produces one token at a time:

```
For output position t = 1, 2, ...:

  1. Render meaning_t from the current Gaussian scene
     → multi-view, multi-pass rendering → a d_f-dimensional vector
  
  2. Decode: project meaning_t onto vocabulary
     logits = meaning_t · [μ₁, μ₂, ..., μ_V]^T
     → dot product with every vocabulary Gaussian's mean
     → highest score = most likely next word
  
  3. Select token_t = argmax(softmax(logits))
  
  4. Activate: look up token_t's Gaussian, add it to the scene
     with positional modulation for position t
  
  5. The scene now has one more Gaussian → repeat
```

This is analogous to rendering a video frame by frame, where each frame adds new objects to the scene (4D Gaussian Splatting). The Gaussian scene grows as generation proceeds.

---

## Component 8: Adaptive Density Control (A7, Training Only)

Every 1,000 training steps, the model examines each Gaussian:

**Split:** If a Gaussian has high gradient (the model wants to move it) AND it's large (broad covariance) → it's trying to represent too much. Split it into two more specific Gaussians. This is how "bank" goes from one broad Gaussian to separate finance/river Gaussians.

**Clone:** If a Gaussian has high gradient but is already small → the region needs more coverage. Clone it and nudge the copy.

**Prune:** If a Gaussian has near-zero opacity → the model decided it's not useful. Remove it.

The vocabulary is not fixed — it grows and shrinks during training, self-organizing to match the data's semantic structure.

---

## Putting It All Together: Full Forward Pass

Input: "The warm coffee sat on the table"

```
1. TOKENIZE: ["The", "warm", "coffee", "sat", "on", "the", "table"]

2. ACTIVATE: Look up each token's Gaussian(s) from vocabulary
   → 7+ Gaussians in 64D splatting space, each carrying 512D features

3. POSITION MODULATE: Shift each Gaussian's parameters based on position
   → Gaussian for "warm" at position 2 gets a displacement encoding "adjective before noun"

4. MULTI-PASS RENDERING (P=8):
   Pass 1: Raw composition — proximity-based blending
   Pass 2: "coffee" pulls "warm" closer (they modify each other)
   Pass 3: "sat" connects to "coffee" (subject-verb)
   ...
   Pass 8: Final refined scene
   
   Each pass: multi-view rendering (8 heads) → 512D meaning vector

5. OUTPUT: The final rendered meaning vector (512D)
   → Decode to vocabulary → predict next token
```

For generation, append predicted token's Gaussian to the scene and repeat.

---

## Model Size Comparison

| Component | Parameters | Equivalent |
|---|---|---|
| Vocabulary (50K Gaussians) | 133M | ~ BERT-base |
| Per-pass MLPs (×8 passes) | 17.3M | ~ 2 transformer layers |
| Viewpoint projections (8 heads) | 4.2M | ~ 1 transformer layer |
| **Total** | **~155M** | ~ BERT-base |

This is a small model by modern standards. The goal isn't to compete with GPT-4 — it's to validate the architectural principle at small scale.

---

*Next: [Article 7 — What Could Go Wrong (And What We'd Learn)](07-risks-and-experiments.md)*
