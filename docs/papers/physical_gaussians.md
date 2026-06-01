# Physical Gaussians: Extending the SGS Primitive with Material Semantics

## Abstract

We propose extending the Semantic Gaussian Splatting (SGS) atomic primitive
with a physical embedding vector that encodes material properties (hardness,
elasticity, friction, density, thermal conductivity). The extended primitive,
which we call a Physical Gaussian, carries three coupled representations:

1. **Semantic embedding** (what it means)
2. **Geometric parameters** (where it is, what shape)
3. **Physical embedding** (how it behaves)

The key insight is that these three are not independent: Gaussians with
similar semantic embeddings and similar covariance structures will naturally
share physical properties. A cluster of small, tightly-packed, high-opacity
Gaussians with the semantic label "stone" will be hard and rigid. A cluster
of large, low-opacity, dispersed Gaussians with the label "cloth" will be
soft and deformable. The physical embedding does not need to be independently
learned for every Gaussian. It can be inferred from the correlation between
semantic and geometric parameters, then refined through physics simulation.

This enables a unified representation where the same scene file that drives
rendering also drives physics, without separate collision meshes, material
assignments, or physics proxies.

## Motivation

### The representation gap in current game engines

Current 3D engines maintain parallel representations:

| Concern | Representation | Format |
|---------|---------------|--------|
| Rendering | Meshes + textures + shaders | .fbx, .gltf |
| Physics | Simplified collision meshes + material tags | PhysX shapes |
| Semantics | Manual labels, gameplay tags | Engine-specific |
| Audio | Surface type enums | Hardcoded |

These are authored separately, maintained separately, and often diverge.
A rock that looks like stone might have the wrong physics material assigned.
A glass window might use a generic "hard surface" collider. The semantic
meaning ("this is glass") exists only in a human's head, not in the data.

### What SGS already provides

SGS represents scenes as collections of Gaussians where each primitive
carries a semantic embedding. After Raum decomposes "a castle on a hill"
into a composition tree, every leaf Gaussian inherits semantic context
from its ancestors: this particular Gaussian is part of a "wall" which is
part of a "castle." The embedding encodes this full context.

### What's missing

The physical behavior. When a cannonball hits the castle wall, what happens?
Currently: nothing, because Gaussians have no physics. To simulate this,
you'd need to export to a mesh, assign physics materials manually, and run
a separate physics engine. The semantic information (this is stone, this is
wood, this is glass) is lost in the export.

## The Physical Gaussian Primitive

### Definition

A Physical Gaussian extends the standard SGS primitive:

```
G = (p, S, q, alpha, c, e_s, e_p, l)

where:
  p     : R^3        — position
  S     : R^3        — scale (log-space, anisotropic)
  q     : R^4        — rotation quaternion
  alpha : R          — opacity (logit)
  c     : R^3        — color (or SH coefficients)
  e_s   : R^d_s      — semantic embedding (d_s = 128-300)
  e_p   : R^d_p      — physical embedding (d_p = 32-64)
  l     : string     — label (human-readable concept name)
```

### Physical embedding dimensions

The physical embedding `e_p` encodes material properties in a learned
continuous space. Unlike discrete material enums ("stone", "wood", "metal"),
it represents a continuous manifold where:

- Nearby points have similar physical behavior
- Interpolation produces physically plausible intermediate materials
- The space is structured by property axes (not arbitrary)

Proposed property axes (interpretable dimensions):

| Axis | Range | Low end | High end |
|------|-------|---------|----------|
| Hardness | [0, 1] | Soft (cloth, foam) | Hard (diamond, steel) |
| Elasticity | [0, 1] | Inelastic (clay) | Elastic (rubber, spring) |
| Friction | [0, 1] | Smooth (ice, glass) | Rough (sandpaper, bark) |
| Density | [0, 1] | Light (air, foam) | Heavy (lead, gold) |
| Brittleness | [0, 1] | Ductile (copper) | Brittle (glass, ceramic) |
| Thermal conductivity | [0, 1] | Insulating (wood) | Conducting (metal) |
| Transparency | [0, 1] | Opaque (stone) | Transparent (glass, water) |
| Deformability | [0, 1] | Rigid (steel) | Deformable (rubber, cloth) |

The remaining dimensions (24-56) are learned and may capture properties
we haven't named: how a material fractures, how it interacts with water,
how it ages, how it sounds when struck.

## The Correlation Hypothesis

### Core claim

Physical properties are not independent of semantic and geometric properties.
They are highly correlated:

**Semantic correlation:**
- "stone" -> hard, heavy, brittle, rough, opaque
- "water" -> soft, elastic, smooth, dense, transparent
- "wood" -> medium hardness, low elasticity, rough, medium density, opaque

**Geometric correlation:**
- Small, tightly-packed, high-opacity Gaussians -> solid, hard surface
- Large, dispersed, low-opacity Gaussians -> gas, fog, soft volume
- Flat, aligned Gaussians (low covariance in one axis) -> sheet, membrane
- Uniform scale, high density -> uniform material (metal, stone)
- Variable scale, clustered -> composite material (concrete, wood grain)

**Covariance-opacity correlation:**
- High opacity + small scale -> particle of a dense material
- Low opacity + large scale -> volumetric effect (smoke, fog, glow)
- High opacity + anisotropic scale -> surface element (wall, floor)

### Implication for learning

The physical embedding does not need to be learned from scratch for every
Gaussian. Given a trained semantic embedding and the geometric parameters,
a small network can predict the physical embedding:

```
e_p = f_phys(e_s, S, alpha, context)
```

where `context` includes the local neighborhood (what's nearby in the
composition tree). This network can be trained on:

1. Synthetic data: assign known physical properties to known materials
2. Physics simulation: run forward simulation on scenes with predicted
   properties, backprop the error when objects don't behave correctly
3. Human annotation: label a small set of Gaussians with physical
   properties, let the network generalize via the semantic/geometric
   correlation

## Collapse Modes

### The unified primitive collapses depending on query

The same Physical Gaussian is queried in different modes depending on
the engine subsystem that needs it:

| Query mode | What's read | Used by |
|------------|-------------|---------|
| Render | p, S, q, alpha, c | Rasterizer (visual appearance) |
| Physics | p, S, q, e_p | Physics engine (collision, dynamics) |
| Semantic | p, e_s, l | AI systems (scene understanding, NPC behavior) |
| Audio | p, e_p, e_s | Sound engine (surface impact sounds) |
| Destruction | p, S, e_p (brittleness) | Fracture simulation |
| Thermal | p, e_p (conductivity) | Heat propagation |

This is analogous to quantum mechanical observables: the Gaussian exists
in a superposition of roles until a specific subsystem "measures" it by
reading the relevant parameters.

### Clustering and material regions

Gaussians that are:
- Spatially proximate (within some radius)
- Semantically similar (cosine similarity of e_s > threshold)
- Geometrically similar (similar S, alpha)

will naturally form **material regions**: contiguous volumes with uniform
physical properties. These regions emerge from the data without explicit
segmentation:

```
material_similarity(G_i, G_j) = w_s * cos(e_s_i, e_s_j)
                                + w_g * exp(-||S_i - S_j||)
                                + w_a * exp(-|alpha_i - alpha_j|)
                                + w_d * exp(-||p_i - p_j|| / r)
```

Where Gaussians above a similarity threshold share physical behavior.
This provides:
- Automatic material segmentation (no manual painting)
- Soft boundaries (gradual transitions between materials)
- Context-dependent behavior (the same "stone" Gaussian behaves
  differently at a wall's edge than at its center)

## Architecture for Raum 2.0

### Empirical validation of the correlation hypothesis

We validated the correlation between semantic embeddings and physical
properties across 77 materials with 8 physical axes. Results:

| Embedding source | Overall R^2 | Best axis (hardness) |
|-----------------|-------------|---------------------|
| GloVe 300d | 0.24 | 0.54 |
| GloVe + geometric proxies | 0.28 | 0.49 |
| Planck 100M (tok_features) | -0.39 | -0.32 |

**Interpretation:** The correlation exists but is scale-limited. GloVe
encodes co-occurrence ("stone" near "hard"), giving R^2 = 0.54 on
hardness. But 300d word vectors + 77 samples + a 128-unit MLP cannot
capture the full physics manifold. Large language models (1B+) that have
internalized physical world knowledge (GaussianProperty, PhysSplat
confirm this) would perform substantially better. The limitation is
embedding quality and training data quantity, not the fundamental
semantic-physics relationship.

### Two-stage architecture (scale-adaptive)

Based on the empirical findings, we propose a two-stage design that
works at current scale and improves with model size:

**Stage 1: Discrete classifier + lookup (current, 100M scale)**

```
e_s -> material_classifier -> class_id -> e_p (lookup table)
```

- Classify into K=50-100 material classes from semantic embedding
- Each class has a curated e_p vector
- Classification accuracy >> regression R^2 for the same embeddings
- Works immediately with existing Planck/GloVe

**Stage 2: Continuous refinement (future, 1B+ scale)**

```
e_s -> class_id -> e_p_base (lookup) -> e_p + residual (MLP)
```

- Lookup provides a strong initialization
- MLP predicts per-Gaussian residual from contextual features
- Trained with physics simulation feedback (differentiable)
- At frontier scale (10B+), the lookup becomes unnecessary

### Training the physical embedding

**Phase 1: Supervised from semantic labels**

Map known material words to physical property vectors:
- Build a lookup table: {"stone": [0.9, 0.1, 0.7, 0.8, 0.8, 0.3, 0, 0.1], ...}
- For each Gaussian with a known label, assign the lookup vector via classifier
- At current scale: discrete lookup (R^2 = 0.54 on hardness axis)
- At Hertz scale: train continuous MLP with residual refinement

**Phase 2: Physics simulation feedback**

- Take a Raum-generated scene with predicted e_p
- Run a differentiable physics simulator (e.g., DiffTaichi, Warp)
- Apply forces (gravity, collisions, impacts)
- Compare behavior to expected (stone shouldn't deform, cloth should)
- Backprop through the simulator to refine e_p

**Phase 3: Emergent properties**

- Let the free dimensions of e_p learn from simulation data
- Properties we didn't explicitly encode may emerge:
  - How fracture patterns propagate through the material
  - How the material interacts with fluids
  - Resonance frequencies (for audio synthesis)

### Integration with composition tree

The composition tree already carries semantic context. Physical embeddings
inherit context the same way colors do:

```
castle_scene (e_p: aggregate of children)
├── castle (e_p: stone-like, hard, heavy)
│   ├── tower (inherits castle's e_p)
│   │   ├── base (stone, full hardness)
│   │   ├── body (stone, weathered)
│   │   └── flag (cloth! different e_p from parent)
│   └── gate (wood + iron, composite e_p)
└── hill (earth, medium hardness, low elasticity)
```

Leaf Gaussians inherit their parent node's physical embedding but can
override it. The flag is cloth despite being a child of a stone tower.
The gate is wood despite being part of a stone castle. The semantic
embedding already distinguishes these, the physical prediction network
uses that distinction.

## Applications

### 1. Physics-native 3D scenes

Generate a scene with Raum, get physics for free. No manual material
assignment. Drop the scene into a physics engine and objects behave
correctly: stone is rigid, water flows, cloth drapes, wood splinters.

### 2. Destruction simulation

When a force exceeds the material's strength (derived from e_p hardness
and brittleness), the composition tree fractures:
- Break the parent-child links at the fracture point
- Disconnected subtrees become independent rigid bodies
- Fracture patterns follow material grain (from e_p anisotropy)

### 3. Sound synthesis (connects to Klang)

Klang already synthesizes audio from Gaussian parameters. Physical
embeddings provide the missing piece: what sound does this material make?
- Impact sound: f(e_p hardness, size, velocity)
- Resonance: f(e_p density, geometry, hollow vs. solid)
- Scraping: f(e_p friction, contact area, relative velocity)

### 4. Material-aware LOD

When reducing detail (LOD), merge Gaussians that share physical properties
first. Two adjacent stone Gaussians can merge safely. A stone Gaussian and
a glass Gaussian should not merge, even if spatially adjacent, because their
physics differ.

### 5. Procedural weathering and aging

Physical embeddings inform how materials age:
- High brittleness + high exposure -> cracking
- High thermal conductivity + weather cycles -> expansion damage
- Low hardness + friction -> wearing/smoothing
- Organic (from e_s) -> growth, decay

## Comparison to Existing Approaches

| Approach | Physics | Semantics | Unified? |
|----------|---------|-----------|----------|
| Traditional mesh + PhysX | Separate colliders | Manual tags | No |
| NeRF + physics proxy | Extracted mesh | None | No |
| 3DGS + particle physics | Per-splat mass | None | No |
| Material Point Method | Grid-based | None | No |
| **Physical Gaussians (ours)** | Embedded per-Gaussian | Embedded per-Gaussian | **Yes** |

The closest related work:
- **PhysGaussian (2024)**: adds physics to 3DGS but without semantic
  embeddings. Materials are assigned manually or from segmentation.
- **Physics-Informed Deformable GS**: simulates deformation but doesn't
  encode material properties in the primitive itself.
- **PAC-NeRF**: physics-aware compositional NeRF, but uses implicit
  representations rather than explicit Gaussian primitives.

Our contribution: the material is not a label on top of the geometry.
It is part of the primitive. The Gaussian IS the material, not just the
shape.

## Formal Guarantees

The Physical Gaussian primitive and its associated operators rest on a set
of mathematical claims (full statements in `docs/proofs/physical_gaussians_math.md`).
Six of these are machine-verified in Lean 4 via the Aristotle prover. Each
compiles with **zero `sorry`** and uses only the standard axioms (`propext`,
`Classical.choice`, `Quot.sound`). Verified 2026-05-31.

| Claim | Statement | Why it matters | Status |
|-------|-----------|----------------|--------|
| **P1** | Σ = R(q)·diag(exp(2S))·R(q)ᵀ is symmetric positive definite for any unit quaternion q and any log-scale S | The covariance of every Physical Gaussian is well-defined, so covariance similarity (P4) and the physics/render collapse operators are meaningful | ✅ Proven |
| **P2** | Material similarity M(·,·) ∈ [0,1] and M(G,G)=1 | Region clustering (Def. 1.6) is well-posed; the similarity threshold τ has a fixed, bounded scale | ✅ Proven |
| **P4** | Covariance similarity Φ(A,B) ∈ (0,1], Φ(A,A)=1, Φ symmetric, for SPD A,B | The covariance term in M is bounded and order-independent | ✅ Proven |
| **P5** | f_phys (feedforward net with continuous activations) is continuous | Small changes in semantic embedding produce small changes in predicted physics: no discontinuous material jumps | ✅ Proven |
| **P8** | Cloning/splitting a Gaussian yields children with M(G_a,G_b) ≥ 1 − (Σ wₖLₖ)·ε | Densification preserves material coherence; children stay in the same material region under bounded perturbation | ✅ Proven |
| **P9** | d_p ≥ ⌈log₂ K⌉ separates K material classes; d_p = O(K) allows continuous interpolation. At K=100, d_p=32 suffices | Justifies the chosen physical-embedding dimension (32–64) | ✅ Proven |
| **P6** | I(e_p ; e_s, S, α) ≥ I(e_p ; e_s): semantic features carry information about physics (the correlation hypothesis) | The core hypothesis enabling prediction over independent learning | Empirical (GloVe R²=0.54 on hardness); no formal proof intended |
| **P3, P7** | Material regions partition V (graph connectivity); collapse operators π_X are idempotent projections | Definitional / standard results; not separately mechanized | Standard |

### What the proofs establish (and what they do not)

The verified claims guarantee the representation is **internally consistent**:
the covariance is always a valid SPD matrix (P1, P4), the similarity metric
that defines material regions is bounded and self-maximal (P2, P4), the
prediction network behaves continuously (P5), densification cannot silently
split a material region (P8), and the chosen embedding width is sufficient
to encode the target material classes (P9).

They do **not** establish that predicted physics are *correct* — that the e_p
vector assigned to "stone" produces the right rigid-body behavior. That is the
content of the correlation hypothesis (P6), which is empirical. The current
evidence (GloVe R²=0.54 on hardness, scale-limited) is a necessary signal, not
a proof. The two-stage architecture is designed precisely so that the
floor (discrete classifier + lookup) holds even where the continuous
hypothesis is weak at 100M scale.

Proof artifacts (Lean source + Aristotle summaries) are under
`docs/proofs/results/P{1,2,4,5,8,9}_dir/`; submission tracking is in
`docs/proofs/aristotle_proof_tracker.md`.

## Research Questions

1. **What dimensionality for e_p?** 32 seems minimal, 64 generous. Do we
   need all 8 named axes or do fewer suffice for game-engine physics?

2. **At what model scale does continuous prediction become viable?**
   Empirically validated: 100M + GloVe gives R^2 = 0.24 (insufficient for
   continuous). Expected: 1B+ with contextual embeddings gives R^2 > 0.6.
   The discrete classifier + lookup works at any scale as a floor.

3. **Differentiable physics integration.** Which simulator? DiffTaichi,
   Nvidia Warp, and Brax are candidates. Can we backprop through collision
   to refine e_p?

4. **Scale of training data.** 77 materials is insufficient for regression
   but sufficient for classification into broad classes. 500+ materials
   with sentence-transformer embeddings is the next target.

5. **Runtime cost.** Does querying e_p add latency to physics ticks?
   Likely negligible (one extra vector read per Gaussian per frame), but
   needs profiling at 500K+ Gaussians.

6. **Classification vs. regression for current scale.** Binary splits
   (hard/soft, rigid/deformable) may capture 80%+ of the physics-relevant
   variance with near-perfect accuracy from GloVe. This is the practical
   path while continuous prediction matures with scale.

## Roadmap (Raum 2.0)

| Phase | Goal | Duration |
|-------|------|----------|
| 2.0.1 | Add e_p field to GaussianParams, build material lookup table | 1 week |
| 2.0.2 | Train prediction MLP: (e_s, S, alpha) -> e_p | 2 weeks |
| 2.0.3 | Integrate with a simple rigid-body simulator | 2 weeks |
| 2.0.4 | Demo: drop a ball on a Raum-generated scene, observe correct physics | 1 week |
| 2.0.5 | Connect to Klang for impact sound synthesis | 2 weeks |
| 2.0.6 | Differentiable physics loop for e_p refinement | 4 weeks |

Total: ~12 weeks for the full stack. Phase 2.0.1-2.0.4 (proof of concept
with rigid bodies) is achievable in 6 weeks.

## Conclusion

The Physical Gaussian extends SGS from a visual representation to a
complete scene representation: one where the same atomic primitive drives
rendering, physics, audio, semantics, and interaction. The key enabler is
the correlation between what something means (e_s), what it looks like
(S, alpha, c), and how it behaves (e_p). These three are not separate
systems layered on top of each other. They are different projections of
the same underlying reality, encoded in one unified primitive.

This is the fundamental thesis of SGS pushed further: if Gaussians are the
atomic primitive of visual representation, they should also be the atomic
primitive of physical representation. A castle is not one blob. It is a
tree of sub-concepts, each carrying full semantic, geometric, and physical
information, bottoming out at thousands of individually meaningful splats
that know what they are, where they are, and how they behave.

## References

Physics-aware Gaussian splatting:
- Xie, T., Zong, Z., Qiu, Y., Li, X., Feng, Y., Yang, Y., & Jiang, C. (2024). PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics. *CVPR*.
- Huang, T., et al. (2025). DreamPhysics: Learning Physical Properties of Dynamic 3D Gaussians with Video Diffusion Priors. *AAAI*.
- Zhang, T., et al. (2024). PhysDreamer: Physics-Based Interaction with 3D Objects via Video Generation. *ECCV*.
- Borycki, P., et al. (2024). GASP: Gaussian Splatting for Physic-Based Simulations.
- Zhao, H., et al. (2025). PhysSplat (Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting). *ICCV*.
- Lee, Jacobson, & Xue (2026). PG-3DGS: Differentiable Physics in 3D Gaussian Splatting Optimization.
- Xu, X., et al. (2025). GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs. *ICCV*.

Material property prediction:
- Li, X., et al. (2023). PAC-NeRF: Physics Augmented Continuum Neural Radiance Fields for Geometry-Agnostic System Identification. *ICLR*.
- Izadyar & Schneider (2025). LLM-Guided Material Inference from Point Cloud Geometry.

Per-Gaussian feature / semantic splatting:
- Ye, M., Danelljan, M., Yu, F., & Ke, L. (2024). Gaussian Grouping: Segment and Edit Anything in 3D Scenes. *ECCV*.
- Qin, M., et al. (2024). LangSplat: 3D Language Gaussian Splatting. *CVPR*.
- Zhou, S., et al. (2024). Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields. *CVPR*.

Audio from Gaussians:
- Bhosale, S., et al. (2024). AV-GS: Audio-Visual Gaussian Splatting. *NeurIPS*.
- Pang, et al. (2025). VibraVerse: A Large-Scale Geometry-Material-Sound Dataset.

Foundations:
- Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *ACM ToG (SIGGRAPH)*, 42(4).
- Max, N. (1995). Optical Models for Direct Volume Rendering. *IEEE TVCG*, 1(2), 99–108.
- Gorshkov, N. (2026). On the Expressiveness of Alpha-Compositing: A Strict Superset of Softmax Attention. *Preprint*.
- de Moura, L., & Ullrich, S. (2021). The Lean 4 Theorem Prover and Programming Language. *CADE*.

*Note: venues/years for several 2024–2026 entries are drawn from the literature review (`docs/papers/physical_gaussians_literature_review.md`) and should be confirmed against the published versions before external submission.*
