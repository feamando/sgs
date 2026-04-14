# Semantic Gaussian Splatting (SGS)

**Radiance Labs** — Applying the 3D Gaussian Splatting rendering equation to language, audio, and 3D scene generation.

## Core Idea

What if words worked like Gaussians in 3D rendering?

In [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), scenes are represented as clouds of overlapping Gaussian distributions, rendered by alpha-compositing them front-to-back. We apply the same rendering equation to language: each word is a Gaussian in semantic space, and sentence meaning is composed via alpha-compositing — transmittance-gated blending where earlier words consume rendering capacity, leaving less for later words.

**Formally proven:** Softmax attention (used in transformers) is a strict subset of alpha-compositing. Every softmax weight vector can be exactly replicated by alpha-compositing, but not vice versa. [Verified in Lean 4](docs/proofs/lean/claim_3_5_softmax_subset_alpha.lean) with zero `sorry` statements.

## Results

### Benchmarks (Phase 1-3)

| Setting | SGS | Softmax | Significance |
|---------|-----|---------|-------------|
| Zero-shot STS-B (Spearman) | **0.707** | 0.697 | +0.010 |
| Few-shot STS-B (5.7K pairs, 3 seeds) | **0.676 ± 0.002** | 0.649 ± 0.009 | p < 0.05, 5x tighter error bars |
| AllNLI (314K pairs) | 0.726 | 0.729 | Tied at scale (Δ = 0.003) |
| SCAN length generalization | 27.2% | 26.6% | GRU decoder drives this, not encoder |

**Pattern:** SGS has a stronger inductive bias (better zero-shot/few-shot), but softmax catches up at scale. Similar to CNN vs. Vision Transformer.

### Radiance Planck (100M Language Model)

100.9M parameter causal language model using SGS rendering. Trained on TinyStories for 3 epochs (~82K steps, RTX 4090).

Generates coherent children's stories with narrative arcs, character consistency, and natural dialogue:

> *"Once upon a time, there was a little girl named Lily. She loved to play outside in the sun and pick flowers. One day, she saw a big, dark cloud in the sky..."*

[Full samples](results/planck_samples.txt) | [Architecture](src/sgs_lm.py) | [Training script](scripts/train_lm.py)

### Radiance Hertz (1B Language Model)

1.03B parameter SGS language model. Training on FineWeb-Edu (10B tokens). Benchmark against TinyLlama/Pythia.

[Training script](scripts/train_hertz.py) — single command: downloads data + trains.

## Formal Proofs (16 Lean 4 Theorems)

All mathematical foundations formally verified in Lean 4 with Mathlib. Zero `sorry` statements. Only standard axioms.

| Proof | What | File |
|-------|------|------|
| Softmax ⊂ Alpha-Compositing | Every softmax weight vector reproducible by alpha-compositing, not vice versa | [claim_3_5](docs/proofs/lean/claim_3_5_softmax_subset_alpha.lean) |
| Weight sum bounded | Rendering weights sum to at most 1 (telescoping identity) | [claim_3_1](docs/proofs/lean/claim_3_1_weights_sum_bounded.lean) |
| Transmittance monotonicity | Transmittance is non-increasing | [claim_3_2](docs/proofs/lean/claim_3_2_monotonic_transmittance.lean) |
| **Two-pass partition** | Blob + word rendering = single-pass rendering | [claim_H1](docs/proofs/lean/claim_H1_two_pass_partition.lean) |
| **T_max expressiveness** | Transmittance cap preserves relative weight expressiveness | [claim_H2](docs/proofs/lean/claim_H2_transmittance_scaling.lean) |
| **Permutation invariance** | Total rendering weight is order-independent | [claim_H4](docs/proofs/lean/claim_H4_permutation_invariance.lean) |

[All 16 proofs](docs/proofs/lean/)

## Research Tracks

### Active

| Track | Name | Status | Description |
|-------|------|--------|-------------|
| **A** | Core SGS | Complete | STS-B, SCAN, NLI benchmarks. Paper v3 (3x orthogonally challenged). |
| **B1** | Planck (100M) | **Trained** | SGS language model on TinyStories. Generates coherent stories. |
| **B1-1** | Hertz (1B) | Training | SGS language model on FineWeb-Edu. Benchmark vs TinyLlama/Pythia. |
| **B2** | Planck 1.1 (H-SGS) | Code ready | Knowledge Splatting: built-in RAG via Gaussian blobs. 3 Lean 4 proofs verified. |
| **C** | Klang (Audio) | Variant B designed | Audio synthesis via Gaussian splatting. Each Gaussian = a sound layer. |
| **D** | Raum (3D) | PoC designed | Text-to-3D scene generation via structural SGS ↔ 3DGS isomorphism. |
| **E** | Zuse (Code) | Stub whitepaper | Code generation where Gaussian variance = implementation variability. |

### Hierarchical SGS — Knowledge Splatting (Track B2)

The newest research direction. Knowledge blobs are Gaussian distributions representing pre-computed patterns from training data. Two-pass rendering: blobs set a semantic backdrop (Pass 1), word tokens add specifics in the remaining transmittance (Pass 2). This is RAG built into the rendering equation — retrieval and generation unified through the same kernel and compositing math.

[Whitepaper](docs/whitepaper/hierarchical_sgs.md) | [Orthogonal challenge](paper/orthogonal_challenge_hsgs.md) | [Code model stub](docs/whitepaper/sgs_code.md)

## Model Naming

Named after Berlin physicists, ascending scale:

| Model | Params | Named After |
|-------|--------|-------------|
| **Planck** | 100M | Max Planck (1858-1947) — quantum theory |
| **Hertz** | 1B | Heinrich Hertz (1857-1894) — electromagnetic waves |
| **Helmholtz** | TBD | Hermann von Helmholtz (1821-1894) — thermodynamics |
| **Einstein** | TBD | Albert Einstein (1879-1955) — relativity |
| **Zuse** | TBD | Konrad Zuse (1910-1995) — first computer (Z3), first programming language (Plankalkul) |

## Quick Start

See [SETUP.md](SETUP.md) for full setup + experiment instructions (Windows + RTX 4090).

```bash
# Clone
git clone https://github.com/feamando/sgs.git && cd sgs

# Install
python -m venv .venv && .venv/Scripts/activate  # or source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Train Planck (100M, ~2 hours on RTX 4090)
python scripts/train_lm.py --data-dir data/tinystories

# Train Hertz (1B, ~3-5 days on RTX 4090)
python scripts/train_hertz.py

# Train Planck 1.1 with Knowledge Splatting (after Planck 1.0)
python scripts/train_planck11.py
```

## Papers

- [Semantic Gaussian Splatting: Alpha-Compositing as a Composition Mechanism for Language](paper/semantic_gaussian_splatting.md) — Full SGS paper (v3, orthogonally challenged 3x)
- [On the Expressiveness of Alpha-Compositing: A Strict Superset of Softmax Attention](paper/theorem_paper.md) — Standalone theorem paper
- [Hierarchical SGS: Knowledge Splatting](docs/whitepaper/hierarchical_sgs.md) — Built-in RAG via Gaussian blobs
- [Klang Variant B: Layer-Based Audio Gaussian Splatting](docs/klang/whitepaper_variant_b.md) — Audio synthesis

## Interactive Visualizer

[Radiance Prisma](prisma/) — see how meaning is rendered. Visualizes Gaussian composition with transmittance, word weights, and SGS vs softmax comparison.

## License

Research project — not commercial. Exploring what's possible and learning how models work.

## Author

Nikita Gorshkov — [Radiance Labs](docs/brand.md)
