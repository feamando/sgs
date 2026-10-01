# Increasing Scene Fidelity in Raum: From 60 Gaussians to Photorealism

## Current State

Raum 1.3 generates ~60 Gaussians per scene via recursive decomposition. The
output is semantically correct (correct objects, spatial relations, hierarchy)
but visually sparse: point clouds, not solid objects. A lighthouse is
recognizable as a vertical cluster, but has no surface detail, texture, or
realistic geometry.

## Target State

Raum 1.4 goal: 5,000-50,000 Gaussians per scene with recognizable solid
objects, correct proportions, surface detail, and basic materials. Not
photorealistic (that requires 500K+), but clearly "3D objects" rather than
"point clouds."

## The Gap

| Metric | Raum 1.3 | Target (1.4) | Photorealistic |
|--------|----------|--------------|----------------|
| Gaussians/scene | 60 | 5,000-50,000 | 500K-5M |
| Depth levels | 1-2 | 3-4 | N/A (flat) |
| Surface quality | Points | Solid shapes | Textured surfaces |
| View-dependent | No | Basic (SH deg 1) | Full (SH deg 3) |
| Materials | Flat color | Per-object color | BRDF |

## Strategy: Progressive Densification in 3 Stages

### Stage 1: Template-based subdivision (60 -> 1,000 Gaussians)

**Idea:** Each leaf Gaussian in the composition tree is not a final primitive
but a "seed" that expands into a learned shape template.

**How it works:**
1. Build a library of GS shape templates from real 3D scans (Objaverse objects
   trained through gsplat). Each template is ~50-200 Gaussians representing
   one semantic concept (tower, wall, tree, rock, water surface).
2. Train a small MLP (template selector): given a Gaussian's semantic label
   (from GloVe embedding) + parent context, select a template and output
   deformation parameters (scale, rotation, color shift).
3. At generation time, each leaf Gaussian gets replaced by its deformed
   template. 60 leaves x 15 avg template size = ~900 Gaussians.

**Training data:**
- Render 91 views of each Objaverse object using known cameras
- Train gsplat on each object (yields .ply with N Gaussians)
- Cluster objects by semantic label (GloVe embedding of class name)
- Each cluster's centroid becomes a template

**Gate:** Expanded scenes show recognizable solid shapes (not just points).

### Stage 2: Gradient-based densification (1,000 -> 5,000 Gaussians)

**Idea:** Apply the original 3DGS densification strategy (Kerbl et al.) to
our generated scenes. Splats with high positional gradients indicate
under-reconstructed areas that need more detail.

**How it works:**
1. Render the scene from multiple viewpoints (8-16 cameras, orbit)
2. Compute a target: either SDS loss from a 2D diffusion model, or
   reconstruction loss against a reference render (retrieved from database)
3. Run 100-200 densification iterations:
   - Clone small splats where gradient > threshold (under-reconstructed area)
   - Split large splats where gradient > threshold (over-covering area)
   - Prune splats where opacity < 0.01 (useless, invisible)
4. Output: denser scene with filled surfaces and fewer gaps

**Key decisions:**
- What is the "target"? Options:
  - (a) SDS from Stable Diffusion (no ground truth needed, but noisy)
  - (b) Multi-view consistency loss (self-supervised, penalizes floaters)
  - (c) Retrieved reference scene (clean signal, limited by retrieval quality)
- Gradient threshold and schedule (start permissive, tighten)

**Gate:** No floating artifacts, shapes are solid from all viewing angles.

### Stage 3: Appearance refinement (5,000 -> 50,000+ Gaussians)

**Idea:** Add surface detail, texture variation, and view-dependent effects.

**Options (choose one after Stage 2 ships):**

**Option A: Score Distillation Sampling (SDS)**
- Use Stable Diffusion / SDXL as a multi-view critic
- For each viewpoint, render the scene, encode into latent space, compute
  SDS gradient, backpropagate to Gaussian params
- Gradually increases detail to match what the diffusion model "expects"
  for that text prompt from that viewpoint
- Pro: no paired training data needed. Con: over-smoothing, Janus problem
- Mitigation: use Interval Score Matching (LucidDreamer) instead of vanilla SDS

**Option B: Retrieval-augmented texturing**
- Index a database of high-resolution GS patches (wall segments, foliage,
  water surfaces, brick textures, metal surfaces)
- For each Gaussian cluster in the scene, find the closest GS patch by
  semantic + geometric similarity
- Deform and compose the retrieved patch onto the scaffold
- Pro: photorealistic patches from real scans. Con: seam artifacts, limited
  patch library

**Option C: Hierarchical generation (Octree-GS style)**
- Represent the scene as an octree. Level 0 = Raum 1.3 skeleton (60 splats).
  Level 1 = Stage 1 templates (1,000). Level 2 = densified (5,000). Level 3+ =
  detail.
- Train a level-conditional generator: given level N, produce level N+1 by
  adding detail to each region.
- Enables LOD rendering natively: web viewers use L1-2, desktop uses L3+.

**Gate:** Human quality evaluation. Rendered 2D views should look like
recognizable 3D objects (not necessarily photorealistic, but clearly "a tower"
not "some dots arranged vertically").

## Training on Real Objects: The Key to Fidelity

The fundamental bottleneck is that Raum 1.3 has never seen what real objects
look like as Gaussian splats. It generates splat parameters from semantic
reasoning alone.

To generate realistic brick walls, the model must train on real brick walls
represented as Gaussian splats. This requires:

1. **Scan -> GS pipeline:** Objaverse objects rendered to multi-view images
   (known cameras, no COLMAP needed) -> gsplat training -> .ply per object
2. **Semantic labeling:** Map each GS reconstruction to its text label
   (already available in Objaverse metadata)
3. **Template extraction:** Cluster and average GS reconstructions per class
4. **Conditional generation:** Train the subdivision/densification MLPs on
   paired (text label, target GS) data

### Priority object categories for training

| Category | Source | Why |
|----------|--------|-----|
| Architectural (walls, towers, roofs) | Objaverse + ABO | Core Raum demo objects |
| Natural (rocks, trees, terrain, water) | Objaverse + CO3D | Scene backgrounds |
| Vehicles (ships, cars) | ShapeNet + Objaverse | Demo scenes |
| Furniture (chairs, tables) | ABO + Google Scanned Objects | Indoor scenes |

## Recommended Roadmap

| Phase | Gaussians | Duration | Depends on |
|-------|-----------|----------|-----------|
| A: Template subdivision | 60 -> 1,000 | 1-2 weeks | Objaverse GS data build |
| B: Gradient densification | 1,000 -> 5,000 | 1 week | Phase A |
| C: Appearance refinement | 5,000 -> 50,000+ | 2-3 weeks | Phase B + choice of method |

Total: 4-6 weeks for the full pipeline. Phase A is the critical path (needs
the GS training data from Objaverse). Phases B and C can iterate in parallel
on different scenes.

## Connection to Raum 1.3

Raum 1.3's recursive decomposition tree is the scaffold. Every stage above
operates on nodes of that tree:
- Stage 1 replaces leaf nodes with templates
- Stage 2 densifies within each node's bounding volume
- Stage 3 adds detail conditioned on the semantic label from the tree

The composition tree structure (parent-child spatial relations) is preserved
at all fidelity levels. This means the DSL editing capability (move the tower,
remove the gate) continues to work at any resolution.

## Open Questions

1. **What's the minimum Objaverse GS training set?** Do we need all 800K
   objects or can 1,000 well-chosen objects cover the semantic space?
2. **SDS vs. retrieval for Stage 3?** SDS needs no data but has quality
   issues. Retrieval needs a good index but produces sharper results.
3. **How to handle articulated objects?** A tree is rigid. A flag waving or
   water flowing needs per-node animation parameters (future work).
4. **View-dependent effects:** At what stage do we add SH coefficients?
   Probably Stage 3, when surface normals become meaningful.
