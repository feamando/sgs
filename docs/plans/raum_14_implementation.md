# Raum 1.4 Implementation Plan: High-Fidelity Scene Generation

## Goal

Scale Raum output from 60 Gaussians (semantic skeleton) to 5,000-50,000+
(solid objects with surface detail). The composition tree from Raum 1.3
remains the scaffold; fidelity is added as post-processing stages.

## Architecture Overview

```
Text prompt
  |
  v
[Raum 1.3 Decomposer] -- existing, no changes
  |
  v
Composition Tree (60 Gaussians, depth 1-2)
  |
  v
[Stage 1: Template Subdivision] -- NEW
  |
  v
Expanded Tree (500-1,000 Gaussians)
  |
  v
[Stage 2: Gradient Densification] -- NEW
  |
  v
Dense Scene (3,000-5,000 Gaussians)
  |
  v
[Stage 3: Appearance Refinement] -- NEW (future)
  |
  v
High-Fidelity Scene (10,000-50,000+ Gaussians)
```

## Prerequisites

- Objaverse GS training data (see `docs/papers/gs_scan_datasets.md`)
- gsplat library installed
- Blender (for multi-view rendering of Objaverse objects)

---

## Phase A: GS Training Data Build

**Duration:** 1-2 days compute (can run overnight)

### A.1 Select Objaverse objects

Pick 500 objects across priority categories:

```
data/objaverse_gs/
  tower/        (50 objects)
  wall/         (50 objects)
  roof/         (30 objects)
  column/       (20 objects)
  arch/         (20 objects)
  rock/         (50 objects)
  tree/         (50 objects)
  terrain/      (30 objects)
  water/        (20 objects)
  ship/         (30 objects)
  car/          (30 objects)
  furniture/    (50 objects)
  misc/         (70 objects)
```

Selection criteria:
- Single objects (not full scenes)
- Clean geometry (no floating parts)
- 1K-50K triangles (not too simple, not too complex)

### A.2 Multi-view rendering

For each object, render 91 views (same as ABO) using Blender:
- Camera on a sphere at 3 elevations (15, 30, 45 degrees)
- 30 azimuth positions per elevation + 1 top-down
- Resolution: 512x512
- White background (easy to mask)
- Output: RGB + depth + camera intrinsics/extrinsics as JSON

Script: `scripts/build_objaverse_gs.py`

### A.3 gsplat training

For each object, train a GS representation:
- 7,000 iterations (fast, sufficient for clean objects)
- Output: .ply file with 1K-10K Gaussians per object
- Store Gaussian params: positions, scales (3), rotations (quat 4),
  opacity, SH degree 0 (RGB only for now)

Script: `scripts/train_gs_objects.py` (batch, loops over all rendered objects)

### A.4 Deliverables

```
data/objaverse_gs/
  {category}/{object_id}/
    renders/          (91 PNG images)
    cameras.json      (intrinsics + extrinsics)
    model.ply         (trained GS, 1K-10K Gaussians)
    metadata.json     (category, name, triangle count, Gaussian count)
```

---

## Phase B: Template Extraction

**Duration:** 2-3 hours

### B.1 Normalize GS representations

All objects need to be in a canonical frame:
- Center at origin
- Fit within unit sphere (max extent = 1.0)
- Align principal axis (PCA on positions)

### B.2 Cluster by category

Within each category, cluster objects by shape similarity
(compare Gaussian position distributions):
- Use Earth Mover's Distance or Chamfer Distance on positions
- k-means within category (k=3-5 templates per category)
- Pick the exemplar (closest to centroid) as the template

### B.3 Template format

```json
{
  "category": "tower",
  "template_id": 0,
  "n_gaussians": 142,
  "gaussians": {
    "positions": [[x,y,z], ...],
    "scales": [[sx,sy,sz], ...],
    "rotations": [[w,x,y,z], ...],
    "opacities": [o, ...],
    "colors": [[r,g,b], ...]
  },
  "bounding_box": {"min": [...], "max": [...]}
}
```

Store at `data/templates/{category}_{template_id}.json`.

---

## Phase C: Subdivision MLP

**Duration:** 3-5 days

### C.1 Model architecture

Input:
- Parent Gaussian params: position (3), scale (1), color (3) = 7
- Semantic embedding: GloVe 300d of the node's label = 300
- Parent context: average of sibling embeddings = 300
- Total input: 607

Output:
- Template selection: softmax over available templates for this category
- Deformation params: position offset (3), scale factor (3),
  color shift (3), rotation delta (4) = 13

Architecture:
- 3-layer MLP: 607 -> 512 -> 256 -> (n_templates + 13)
- ReLU activations, LayerNorm between layers
- ~500K parameters (tiny, fast inference)

### C.2 Training data

For each (category, template) pair, create training samples:
- Input: a "parent" Gaussian representing the object at low resolution
  (single Gaussian at the object's center, with the object's average color)
- Target: the template's full Gaussian set (normalized to parent frame)
- Augment: random scale, rotation, color jitter on the parent

### C.3 Training

- Optimizer: AdamW, lr=1e-3, cosine decay
- Loss: Chamfer distance between predicted and target Gaussian positions
  + L1 on color + L1 on scale
- Epochs: 200 (small dataset, fast convergence)
- Batch size: 64

### C.4 Inference

```python
for leaf in tree.leaves():
    category = classify_label(leaf.name)  # GloVe nearest-neighbor
    template_id, deformation = subdivision_mlp(leaf, category)
    template = load_template(category, template_id)
    expanded = apply_deformation(template, deformation, leaf)
    leaf.replace_with(expanded)
```

---

## Phase D: Gradient-Based Densification

**Duration:** 1 week

### D.1 Multi-view rendering setup

Render the subdivided scene from 16 viewpoints (orbit):
- 8 azimuth x 2 elevations
- Resolution: 256x256 (fast, sufficient for gradient signal)

### D.2 Loss function

Options (implement both, A/B test):

**Option 1: Self-supervised multi-view consistency**
- Render from view A, warp to view B using depth, compare
- Penalizes floaters and gaps (they appear/disappear across views)
- No external model needed

**Option 2: SDS from Stable Diffusion**
- Render from random view, encode to latent, compute SDS gradient
- Provides "what should this look like" signal
- Requires diffusion model on GPU (memory pressure)

Start with Option 1 (no extra model needed).

### D.3 Densification loop

```python
for iteration in range(200):
    renders = render_multiview(scene, cameras_16)
    loss = multiview_consistency_loss(renders)
    loss.backward()

    for gaussian in scene.gaussians:
        if gaussian.position_grad.norm() > threshold:
            if gaussian.scale.max() > scale_threshold:
                split(gaussian)  # too big, divide
            else:
                clone(gaussian)  # too sparse, duplicate nearby

    prune(scene, opacity_threshold=0.01)
    optimizer.step()
```

### D.4 Expected behavior

- Iteration 1-50: fill obvious gaps (undersides, backs of objects)
- Iteration 50-100: refine edges and boundaries
- Iteration 100-200: diminishing returns, stabilize

---

## Phase E: Integration with Raum 1.3 Demo

**Duration:** 2-3 days

### E.1 Pipeline integration

The existing `infer_decomposer.py --serve` web UI gets a fidelity toggle:
- "Skeleton" (L0): raw decomposer output, 60 Gaussians (instant)
- "Templates" (L1): after subdivision, 500-1000 Gaussians (~2s)
- "Dense" (L2): after densification, 3000-5000 Gaussians (~30s)

### E.2 Caching

Densification is slow. Cache results:
- Hash the composition tree JSON
- If cached .json exists in `cache/`, load instead of recomputing
- Clear cache on model update

---

## Milestones and Gates

| Milestone | Deliverable | Gate |
|-----------|-------------|------|
| Data build | 500 objects as .ply | > 80% objects train successfully |
| Templates | 50+ templates across categories | Visual inspection: shapes recognizable |
| Subdivision MLP | Trained model | Chamfer < 0.05 on held-out objects |
| Subdivision demo | Castle scene at L1 | Tower/wall/gate are solid, not points |
| Densification | Castle scene at L2 | No floaters, solid from all angles |
| Product demo | Raum 0.4 web UI | User types prompt, gets solid 3D scene |

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Objaverse objects too noisy for clean GS | Bad templates | Pre-filter by triangle count and manifold check |
| Subdivision MLP overfits to castle-like scenes | Poor generalization | Diverse training categories, augmentation |
| Densification diverges (Gaussians fly away) | Broken scenes | Bounding box constraint, gradient clipping |
| SDS OOMs alongside scene on 24 GB | Can't use Option 2 | Default to multi-view consistency (Option 1) |
| Templates don't deform well | Stiff/wrong shapes | Allow template blending (weighted average of top-2) |
