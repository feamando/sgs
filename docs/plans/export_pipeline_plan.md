# Export Pipeline Plan: GS Scenes to 3D Tools

## Problem

Raum outputs scenes as JSON composition trees + flat Gaussian parameter lists.
These work in our custom Three.js viewer but cannot be imported into
professional 3D tools (Unreal Engine, Unity, Blender) or shared via industry
standard formats (USD, glTF, FBX).

## Target Integrations

| Tool | Priority | Method | Format |
|------|----------|--------|--------|
| Unreal Engine 5 | High | XV3DGS plugin (Apache 2.0) | .ply |
| Three.js (web) | High | .splat compressed format | .splat |
| Unity | Medium | UnityGaussianSplatting | .ply |
| Blender | Medium | Community add-on | .ply point cloud |
| Standard mesh pipeline | Medium | SuGaR mesh extraction | .obj, .fbx |
| USD (industry standard) | Low | Via mesh conversion | .usd |
| glTF/GLB (web standard) | Low | Via mesh conversion | .glb |

## Format Details

### .ply (primary export)

The standard Gaussian Splatting format. Per-Gaussian attributes:
- Position: float32 x 3 (xyz)
- Scale: float32 x 3 (log-scale)
- Rotation: float32 x 4 (quaternion)
- Opacity: float32 x 1 (logit, pre-sigmoid)
- SH coefficients: float32 x 48 (degree 3, 16 per RGB channel)
- Total: ~250 bytes per Gaussian

For Raum 1.3 (60 Gaussians): ~15 KB per scene.
For Raum 1.4 target (50K Gaussians): ~12 MB per scene.

### .splat (web export)

Compressed format for fast web delivery:
- Position: float32 x 3
- Scale: float32 x 3
- Color: uint8 x 4 (RGBA, no SH, single viewpoint)
- Rotation: uint8 x 4 (compressed quaternion)
- Total: ~32 bytes per Gaussian

60 Gaussians: ~2 KB. 50K Gaussians: ~1.6 MB.

### Mesh conversion (SuGaR)

For tools that require traditional meshes:
1. Regularize Gaussians to align with surfaces
2. Poisson surface reconstruction from Gaussian centers + normals
3. UV unwrap + bake Gaussian colors to texture
4. Output: .obj with .mtl + texture atlas

Quality is high for solid objects but degrades for volumetric effects
(smoke, clouds, foliage). Best suited for architectural and hard-surface
objects.

## Implementation Plan

### Phase 1: .ply export (1-2 days)

Write `scripts/export_ply.py` that:
1. Loads a Raum scene JSON (composition tree)
2. Flattens to world-space Gaussians via `tree_to_tensors()`
3. Adds rotation quaternions (currently implicit, derive from tree hierarchy)
4. Writes standard .ply binary format

Validation: import into XV3DGS UE5 plugin, verify rendering matches
our Three.js viewer.

```python
# Pseudo-API
from src.raum.decomposition import load_tree, tree_to_tensors
from src.export import write_ply

tree = load_tree("output/scene.json")
tensors = tree_to_tensors(tree)
write_ply(tensors, "output/scene.ply")
```

### Phase 2: .splat web export (1 day)

Write `scripts/export_splat.py` that:
1. Reads .ply file
2. Evaluates SH at a fixed viewpoint to get RGB
3. Compresses quaternions to uint8
4. Writes .splat binary

Validation: load in a standard web viewer (e.g., antimatter15/splat).

### Phase 3: SuGaR mesh extraction (1 week)

Integrate SuGaR (or GOF) for mesh extraction:
1. Regularize Gaussians (flatten to surface-aligned disks)
2. Run Poisson reconstruction
3. Bake texture from Gaussian colors
4. Export .obj + .mtl + .png texture atlas

This is the most complex phase. Consider using the SuGaR library directly
rather than reimplementing.

```powershell
# Using SuGaR (if installed)
python -m sugar.extract_mesh `
  --scene output/scene.ply `
  --output output/scene_mesh.obj `
  --resolution 512
```

### Phase 4: UE5 integration test (2-3 days)

1. Install XV3DGS plugin in UE5
2. Import .ply from Phase 1
3. Verify: rendering quality, LOD, hybrid rendering with UE meshes
4. Test: import SuGaR mesh from Phase 3 as Nanite mesh
5. Document pipeline in a how-to guide

## Current Gaps

### Rotation representation

Raum's `CompositionNode` stores rotation as Euler angles `[rx, ry, rz]`.
The .ply format expects quaternions. Need to add Euler-to-quaternion
conversion in the export path.

### SH coefficients

Raum currently stores only flat RGB color per Gaussian. No spherical
harmonics. For .ply export, we need to decide:
- Option A: Set SH degree 0 only (flat color). Simplest, works everywhere.
- Option B: Compute SH degree 1 from scene lighting assumptions. Better
  quality in viewers that use SH.

Recommendation: Start with degree 0 (Phase 1-2), add degree 1 in a later
iteration when appearance refinement (Raum 1.4 Stage 3) adds real
view-dependent data.

### Scale representation

Raum stores `scale` as a single float (uniform). .ply format expects
3 log-scale values (anisotropic). Export should write `[log(s), log(s), log(s)]`
for uniform splats, with anisotropic support added when the model learns
non-uniform scales.

## Dependencies

- Phase 1-2: no external dependencies beyond numpy/struct
- Phase 3: SuGaR library (MIT license), requires gsplat
- Phase 4: Unreal Engine 5.1+, XV3DGS plugin (Apache 2.0)

## Success Criteria

1. A Raum-generated scene can be opened in UE5 within 5 minutes
2. Web export (.splat) loads in any standard splat viewer
3. Mesh export (.obj) imports into Blender/Maya with correct geometry
4. Round-trip fidelity: exported scene visually matches our Three.js viewer
