# Radiance Zuse: Gaussian Splatting for Code Generation

**Radiance Labs — Research Proposal (Stub)**
**Authors:** Nikita Gorshkov
**Date:** 2026-04-14
**Status:** Early Concept — Pending H-SGS (Knowledge Splatting) validation
**Depends on:** SGS core, Planck 1.1 results (Track B2 validation)

---

## Why "Zuse"?

**Konrad Zuse** (1910-1995) — born in Berlin-Wilmersdorf. Built the Z3 (1941), the world's first programmable, fully automatic computer. Invented Plankalkül (1945), the first high-level programming language. Filed his first computer patent at age 26. Literally the inventor of modern computing, and a Berliner.

The Zuse model line applies SGS principles to code generation — computation as rendering.

---

## 1. The Problem: Why Code Needs a Different SGS

Hierarchical SGS (Knowledge Splatting) works for natural language because:
- Meaning is continuous and fuzzy → Gaussians model semantic breadth
- Composition is gradual → transmittance naturally handles emphasis
- Similarity is a spectrum → kernel evaluation gives soft relevance scores

Code breaks these assumptions:
- **Syntax is discrete and exact.** `import numpy` is a point, not a distribution. A missing parenthesis is wrong, not "approximately right."
- **Semantic breadth doesn't apply to tokens.** The word "function" in English has many senses; the keyword `def` in Python has exactly one.
- **Composition is structural, not sequential.** A function body must be syntactically enclosed in a function definition. Transmittance (sequential front-to-back) doesn't model nesting.

Applying standard SGS blobs (where Σ = semantic breadth) to code would produce blobs with near-zero variance for everything — collapsing to point lookups. This isn't wrong, but it wastes the Gaussian framework.

---

## 2. Core Insight: Variance as Implementation Variability

In natural language SGS:
- **μ** = what a word means
- **Σ** = how broad that meaning is (polysemy, ambiguity)

In code SGS (Zuse):
- **μ** = the canonical implementation pattern
- **Σ** = how much variation exists across implementations of the same intent

Examples:

| Intent | μ (canonical) | Σ (variability) | Why |
|--------|---------------|-----------------|-----|
| Sort an array | `sorted(arr)` | **High** | Quicksort, mergesort, timsort, key functions, reverse, in-place... |
| Import numpy | `import numpy as np` | **Near-zero** | Only one standard form |
| Build a dashboard | Flask + Chart.js + CSS grid | **Very high** | React/Vue/Svelte, D3/Chart.js/Plotly, REST/GraphQL... |
| Read a CSV file | `pd.read_csv(path)` | **Medium** | pandas, csv module, polars, different params |
| HTTP GET request | `requests.get(url)` | **Medium** | requests, urllib, httpx, aiohttp, error handling varies |

This reinterpretation preserves the Gaussian framework but gives Σ a useful meaning for code: **how many valid ways exist to accomplish this intent.**

### 2.1 What This Enables

- **High-Σ blobs** (many implementations) → the model has more freedom in rendering, adapting to context
- **Low-Σ blobs** (one right way) → the model is strongly constrained, reducing errors
- **Kernel evaluation** K(query, blob) now means: "how relevant is this implementation pattern to the current generation context?"
- **Transmittance** means: "this pattern has already been applied, subsequent patterns should fill in remaining gaps"

### 2.2 Blob Structure for Code

```
CodeBlob = {
  μ_b  ∈ ℝ^{d_s}    — canonical implementation embedding
  Σ_b  ∈ ℝ^{d_s}    — implementation variability (how many ways to do this)
  α_b  ∈ (0, 1)      — pattern reliability (well-tested = high, rare = low)
  f_b  ∈ ℝ^{d_f}     — feature vector (semantic content of the pattern)
  
  -- Code-specific metadata (not used in forward pass, used for traceability):
  template: str       — canonical code template
  language: str       — programming language
  domain: str         — "web", "data", "systems", "ml", etc.
  imports: list[str]  — required imports for this pattern
}
```

---

## 3. Architecture Differences from H-SGS

| Aspect | H-SGS (Language) | Zuse (Code) |
|--------|------------------|-------------|
| Gaussian Σ means | Semantic breadth | Implementation variability |
| Blob source | Text chunks (clustered) | Code patterns (mined from repos) |
| Composition model | Sequential (transmittance) | Structural (AST-aware) — open question |
| Evaluation | Perplexity, human eval | pass@k (HumanEval, MBPP), execution tests |
| Tokenization | BPE on natural language | BPE on code (different token distribution) |
| Training data | TinyStories, FineWeb-Edu | The Stack, GitHub code |

### 3.1 Open Architecture Questions

**Q1: Should the rendering equation change for code?**

The sequential front-to-back rendering may not suit code's tree structure. A function is not a sequence of tokens — it's a tree (AST). Options:
- **Keep sequential rendering** and rely on the autoregressive decoder to handle structure. (Simplest — this is what all current code LLMs do.)
- **Tree-structured rendering** where Gaussians are composited in AST order. (Novel but complex.)
- **Hybrid**: sequential for token generation, tree-structured for blob composition.

Recommendation: Start with sequential (same as H-SGS). If results are promising, explore tree-structured as a follow-up.

**Q2: How are code blobs mined?**

Unlike natural language, code has explicit structure. Blob mining can be:
1. **AST-based**: extract common function patterns, class templates, import patterns from parsed ASTs.
2. **Embedding-based**: embed code chunks with a code encoder (CodeBERT, StarEncoder), cluster by similarity.
3. **Execution-based**: group code snippets that produce the same output for the same inputs (semantic equivalence).

Method 1 is most practical for initial experiments. Method 3 is the most rigorous but expensive.

**Q3: How does Σ get computed from training data?**

For a given intent (e.g., "sort an array"):
1. Mine all implementations from training data that accomplish this intent.
2. Embed each implementation using the code encoder.
3. μ_b = centroid of implementation embeddings.
4. Σ_b = variance of implementation embeddings.
5. High variance = many valid approaches. Low variance = one canonical form.

Challenge: determining "same intent" requires either explicit labels (function docstrings) or automated intent clustering. This is an open research problem.

---

## 4. Proposed Model Line

Following the Radiance Labs naming convention (Berlin scientists, ascending scale):

| Model | Parameters | Training Data | Purpose |
|-------|-----------|---------------|---------|
| **Zuse Z3** | ~100M | The Stack (Python subset, ~5B tokens) | Validate SGS for code, basic completions |
| **Zuse Z4** | ~1B | The Stack (multi-language, ~50B tokens) | Benchmark against StarCoder, CodeLlama |
| **Zuse Plankalkül** | ~3B+ | Full Stack + synthetic | Production-quality code generation |

The Z3/Z4 names reference Zuse's actual computers. Plankalkül references his programming language.

---

## 5. Experiment Plan (Deferred)

### Phase 1: Validate Base Code SGS (Zuse Z3)
**Prerequisites:** Planck 1.1 (H-SGS) validates blob architecture.

1. Train a 100M SGS-LM on Python code (The Stack Python subset).
2. Evaluate on HumanEval (pass@1, pass@10) WITHOUT blobs — baseline.
3. Add code blobs (top 5K function patterns from training data).
4. Evaluate WITH blobs — does retrieval of code patterns improve pass@k?

**Hard gate:** If base SGS on code (no blobs) produces <5% pass@1 on HumanEval, the architecture needs fundamental changes before blob experiments.

### Phase 2: Implementation Variability (Zuse Z3 + Σ)
1. Mine implementation clusters from training data.
2. Compute μ_b and Σ_b for each cluster.
3. Test: does high-Σ blob retrieval produce more diverse completions (higher pass@k for large k)?
4. Test: does low-Σ blob retrieval produce more precise completions (higher pass@1)?

### Phase 3: Scale (Zuse Z4)
Deferred until Phase 1-2 validate the approach.

---

## 6. Risks

| Risk | Severity | Notes |
|------|----------|-------|
| SGS rendering equation doesn't suit code structure | High | Sequential compositing may not model AST nesting. Mitigated by starting simple. |
| Variance-as-variability is a reinterpretation, not a proof | Medium | Need to verify that Σ learned from implementation clusters actually correlates with implementation diversity. Empirical. |
| Code tokenization may not produce meaningful Gaussians | Medium | Code tokens are more discrete than NL tokens. The continuous Gaussian assumption may break down. |
| HumanEval baseline too low | Medium | 100M models typically score <10% on HumanEval. May need larger scale to see meaningful results. |
| This is premature | Low | Intentionally deferred until H-SGS validates the blob architecture on language first. |

---

## 7. Relationship to Other Tracks

```
Track A: Core SGS (STS-B, SCAN, NLI) ─────── Foundation
Track B1: Planck/Hertz (base LMs) ─────────── SGS can generate language ✓
Track B2: H-SGS / Knowledge Splatting ──────── Blobs improve generation (testing)
Track C: Klang (audio) ────────────────────── SGS for audio
Track D: Raum (3D scenes) ─────────────────── SGS for 3D
Track E: Zuse (code) ──────────────────────── SGS for code (this document)
```

Zuse depends on B2 validation: if Knowledge Splatting works for natural language, the blob architecture is sound, and we can adapt it for code with the variance reinterpretation. If B2 fails, Zuse is deferred until we understand why.

---

## 8. References

- Li, Y. et al. (2023). "StarCoder: May the Source Be with You!" arXiv.
- Roziere, B. et al. (2023). "Code Llama: Open Foundation Models for Code." arXiv.
- Chen, M. et al. (2021). "Evaluating Large Language Models Trained on Code." (Codex/HumanEval)
- Feng, Z. et al. (2020). "CodeBERT: A Pre-Trained Model for Programming and Natural Languages."
- Zuse, K. (1972). "Der Computer – Mein Lebenswerk." Springer. (Autobiography)
- Gorshkov, N. (2026). "Hierarchical SGS: Knowledge Splatting." (Track B2)
