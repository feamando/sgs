# Export Pipeline Implementation Plan

## Goal

Enable Raum scenes to be used in Unreal Engine 5, Unity, Blender, and
web viewers. Currently scenes exist only as JSON composition trees
renderable in our custom Three.js viewer.

## Current State: What We Have

`src/raum/decomposition.py` defines:
- `GaussianParams`: position (3), scale (3, log), opacity (1), color (3)
- `CompositionNode`: hierarchical tree, flattens to list of `GaussianParams`
- `tree_to_tensors()`: produces `means`, `scales_log`, `opacities`, `colors`

**Missing for standard .ply export:**
- Rotation (quaternion): currently not stored. All Gaussians are isotropic
  spheres. Need to add `rotation: list[float]` (w, x, y, z) to `GaussianParams`.
- SH coefficients: currently flat RGB only. Need to add `sh_coeffs` field
  for view-dependent color (degree 0 = flat color is fine for now).

## What Are Quaternions and SH Coefficients?

### Quaternions

A quaternion `(w, x, y, z)` represents a 3D rotation without gimbal lock.
In 3DGS, each Gaussian is an ellipsoid (not a sphere), so it needs an
orientation to know which direction it's stretched. The rotation quaternion
combined with the 3-axis scale defines the full 3D covariance matrix:

```
Covariance = R @ diag(s^2) @ R^T
```

where R is the rotation matrix from the quaternion and s is the 3-axis scale.

**Where it lives in SGS:**
- `GaussianParams.rotation: list[float]` = [w, x, y, z], default [1, 0, 0, 0] (identity)
- The subdivision MLP (Raum 1.4) will learn to output non-identity rotations
- Until then, all Gaussians remain spherical (identity quaternion)

### Spherical Harmonics (SH) Coefficients

SH encode how a Gaussian's color changes depending on viewing angle:
- Degree 0: 1 coefficient per channel (3 total) = flat color (what we have now)
- Degree 1: 4 coefficients per channel (12 total) = basic directional variation
- Degree 2: 9 coefficients per channel (27 total) = moderate view-dependence
- Degree 3: 16 coefficients per channel (48 total) = full (standard 3DGS)

**Where it lives in SGS:**
- `GaussianParams.sh_coeffs: list[float]` = SH band coefficients
- Degree 0 only: `sh_coeffs = [r, g, b]` (equivalent to current `color` field)
- Higher degrees added in Raum 1.4 Phase C (appearance refinement)

---

## Implementation Phases

### Phase 1: Extend GaussianParams (0.5 days)

Update `src/raum/decomposition.py`:

```python
@dataclass
class GaussianParams:
    position: list[float]       # [x, y, z]
    scale: list[float]          # [sx, sy, sz] log-scale
    rotation: list[float]       # [w, x, y, z] quaternion, default identity
    opacity: float              # logit (pre-sigmoid)
    color: list[float]          # [r, g, b] in [0, 1]
    sh_degree: int = 0          # 0 = flat color only
    sh_coeffs: list[float] | None = None  # higher-degree SH, None = use color as DC
```

Default rotation = `[1, 0, 0, 0]` (identity quaternion, spherical Gaussian).
Default sh_degree = 0 (flat color, backward compatible).

Update `flatten_gaussians()` to propagate rotation through the tree
(child rotation = parent rotation * child local rotation).

Update `to_dict()` / `from_dict()` for JSON serialization.

### Phase 2: .ply exporter (1 day)

Create `scripts/export_ply.py` and `src/export/ply.py`:

```python
def write_ply(tensors: dict, path: str, sh_degree: int = 0):
    """
    Write standard 3DGS .ply file.

    Header format (binary_little_endian):
      property float x, y, z           (position)
      property float nx, ny, nz        (normals, zero-filled)
      property float f_dc_0..2         (SH degree 0 = RGB)
      property float f_rest_0..N       (higher SH bands, if any)
      property float opacity           (logit)
      property float scale_0..2        (log-scale)
      property float rot_0..3          (quaternion wxyz)
    """
```

PLY uses a specific property ordering that tools expect. Must match exactly
for XV3DGS / UnityGaussianSplatting / gsplat viewers to load it.

### Phase 3: .splat exporter (0.5 days)

Create `src/export/splat.py`:

```python
def write_splat(tensors: dict, path: str):
    """
    Write compressed .splat format (32 bytes per Gaussian).

    Layout per Gaussian:
      float32 x, y, z        (12 bytes)
      float32 sx, sy, sz     (12 bytes, NOT log)
      uint8 r, g, b, a       (4 bytes, color + opacity)
      uint8 qx, qy, qz, qw  (4 bytes, compressed quaternion)
    """
```

Evaluate SH at a fixed viewpoint (front-facing) to compute the single RGB.
Compress quaternion to uint8: map [-1, 1] to [0, 255].

### Phase 4: Mesh exporter via SuGaR (1 week)

Create `scripts/export_mesh.py`:

Two approaches (implement simpler one first):

**Approach A: Poisson reconstruction (no SuGaR dependency)**
1. Extract Gaussian positions as point cloud
2. Estimate normals from local neighborhood (PCA on 10 nearest)
3. Run Open3D Poisson reconstruction
4. Simplify mesh to target face count
5. Bake vertex colors from nearest Gaussian

```python
import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(positions)
pcd.normals = o3d.utility.Vector3dVector(normals)
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
```

**Approach B: SuGaR (higher quality, heavier dependency)**
1. Install SuGaR library
2. Run their regularization + extraction pipeline
3. Output .obj with UV-mapped texture

Start with Approach A (simpler, fewer dependencies). Upgrade to B if
quality is insufficient for UE5/Unity.

### Phase 5: UE5 integration (2-3 days)

1. Install XV3DGS plugin (Apache 2.0, UE 5.0+)
2. Import .ply from Phase 2 -> verify renders in-engine
3. Import .obj from Phase 4 -> enable Nanite, verify LOD
4. Test hybrid rendering: GS splats + standard UE meshes in same scene
5. Write `docs/guides/ue5_import.md` with screenshots

---

## File Structure

```
src/export/
  __init__.py
  ply.py            (write_ply, read_ply)
  splat.py          (write_splat)
  mesh.py           (extract_mesh_poisson, extract_mesh_sugar)
  utils.py          (euler_to_quaternion, sh_eval, normalize_quaternion)

scripts/
  export_ply.py     (CLI: scene.json -> .ply)
  export_splat.py   (CLI: .ply -> .splat)
  export_mesh.py    (CLI: .ply -> .obj)

tests/
  test_export_ply.py
  test_export_splat.py
  test_export_mesh.py
  test_export_roundtrip.py
```

## Dependencies

| Phase | New dependencies |
|-------|-----------------|
| 1-2 | numpy, struct (stdlib) |
| 3 | numpy, struct (stdlib) |
| 4A | open3d |
| 4B | sugar (optional) |
| 5 | Unreal Engine 5.1+, XV3DGS plugin |

## Testing Strategy

### Unit tests

- `.ply` header matches expected property list
- Byte count = header_size + N * bytes_per_gaussian
- Quaternions are normalized (||q|| = 1.0 within 1e-5)
- Round-trip: write .ply -> read .ply -> positions match input

### Integration tests

- Load exported .ply in gsplat viewer (Python): renders without error
- Load .splat in web viewer: displays correctly
- Load .obj in Blender (scripted): vertex count > 0, no degenerate faces

### Visual validation

- Export castle scene -> screenshot from fixed camera -> compare to
  Three.js viewer screenshot (SSIM > 0.8, accounting for renderer differences)

## Timeline

| Phase | Duration | Depends on |
|-------|----------|-----------|
| 1: Extend GaussianParams | 0.5 days | Nothing |
| 2: .ply exporter | 1 day | Phase 1 |
| 3: .splat exporter | 0.5 days | Phase 2 |
| 4: Mesh exporter | 1 week | Phase 2 |
| 5: UE5 integration | 2-3 days | Phase 2 + UE5 installed |

Phases 4 and 5 can run in parallel. Total: ~2 weeks.
