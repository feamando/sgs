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

### Training the physical embedding

**Phase 1: Supervised from semantic labels**

Map known material words to physical property vectors:
- Build a lookup table: {"stone": [0.9, 0.1, 0.7, 0.8, 0.8, 0.3, 0, 0.1], ...}
- For each Gaussian with a known label, assign the lookup vector
- Train a small MLP to predict e_p from (e_s, S, alpha)
- The MLP learns the correlations

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

## Research Questions

1. **What dimensionality for e_p?** 32 seems minimal, 64 generous. Do we
   need all 8 named axes or do fewer suffice for game-engine physics?

2. **Can e_p be fully predicted from (e_s, S, alpha)?** Or does it need
   independent supervision? The correlation hypothesis says most of it is
   predictable, but edge cases (painted metal that looks like wood) may need
   explicit annotation.

3. **Differentiable physics integration.** Which simulator? DiffTaichi,
   Nvidia Warp, and Brax are candidates. Can we backprop through collision
   to refine e_p?

4. **Scale of training data.** How many material-labeled Gaussians do we
   need? Hypothesis: 50-100 canonical materials with known properties,
   the network generalizes via semantic correlation.

5. **Runtime cost.** Does querying e_p add latency to physics ticks?
   Likely negligible (one extra vector read per Gaussian per frame), but
   needs profiling at 500K+ Gaussians.

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
