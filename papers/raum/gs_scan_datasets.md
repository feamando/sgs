# Available Gaussian Splatting Scan Datasets and Resources

## Pre-trained Gaussian Splat Datasets

These provide ready-to-use .ply files with trained Gaussian parameters.

| Dataset | Size | Format | License | Notes |
|---------|------|--------|---------|-------|
| Voxel51/gaussian_splatting | 3.49 GB | .ply + PNG | Apache 2.0 | 4 scenes (drjohnson, playroom, train, truck); 7K and 30K iteration reconstructions |
| DsnTgr/gaussian-splatting | 1.79 GB | JPG/PNG | Public | 920 multi-view images for GS training |
| GaussianSplattingBogota (niduque) | Unknown | Unknown | Unknown | Urban scenes (333 likes on HuggingFace) |
| Physics-Informed-Deformable-GS | Unknown | Unknown | Unknown | Deformable scenes (831 likes on HuggingFace) |

Pre-trained .ply splat files remain scarce. Most available data is multi-view
images that require the full training pipeline.

## Major 3D Datasets Convertible to Gaussian Splats

| Dataset | Scale | Format | License | Pipeline |
|---------|-------|--------|---------|----------|
| **Objaverse-XL** | 10M+ objects | Blender meshes | Varies per object | Render multi-view -> gsplat train |
| **Objaverse 1.0** | ~800K objects | glTF, OBJ | CC-BY 4.0 (mostly) | Render multi-view -> gsplat train |
| **ShapeNet** | ~51K models, 55 categories | OBJ, mesh | Research-only | Render -> gsplat |
| **ABO (Amazon Berkeley Objects)** | 7,900 3D models | glTF 2.0, 4K textures | CC BY 4.0 | 91-view renders already provided |
| **Google Scanned Objects** | ~1,000 household objects | Meshes + textures | CC-BY 4.0 | High-quality photorealistic |
| **CO3D (Meta)** | ~19K sequences, 50 categories | Multi-view + point clouds | CC BY-NC 4.0 | Has pre-computed poses |
| **ScanNet++** | 1,500+ indoor scenes | RGB-D, meshes, point clouds | Research-only | Pre-computed poses |
| **Mip-NeRF 360** | 9 scenes | Multi-view + COLMAP | Research use | Standard GS benchmark |
| **Tanks and Temples** | 21 scenes | Video + GT point clouds | Free for benchmarking | Complex outdoor/indoor |
| **DTU** | 124 scenes, 49-64 views each | 1600x1200 images + point clouds | Free (cite paper) | 338 GB total |

## Object Categories Available

### Architecture and building materials
- **Objaverse:** thousands of building models, towers, castles, walls
- **ABO:** furniture with architectural context
- **ScanNet++:** real interior walls, floors, ceilings
- **Bricks/stone specifically:** limited dedicated datasets. Custom scanning
  recommended, or extract patches from ScanNet++ reconstructions.

### Natural objects
- **Objaverse:** rocks, trees, terrain, plants
- **CO3D:** outdoor objects (plants, hydrants, benches, bicycles)
- **Mip-NeRF 360:** garden, flowers, bicycle scenes

### Vehicles
- **ShapeNet:** cars (7,497), airplanes (4,045), boats, buses
- **Objaverse:** thousands of vehicles at varying quality
- **CO3D:** real cars, motorcycles

### Furniture and household
- **ABO:** 7,900 models with 4K textures, best quality
- **Google Scanned Objects:** ~1,000 household items, photorealistic
- **ShapeNet:** chairs (6,778), tables (8,509), sofas, lamps

## Recommended Pipeline: Scan to Training Data

### Fastest path (synthetic, no COLMAP needed)

```
Objaverse/ABO mesh -> Blender render (91 views, known cameras)
  -> gsplat train (no COLMAP, cameras are exact)
  -> .ply (trained GS, 5K-50K Gaussians per object)
  -> Semantic label from metadata
  -> (label, GS) training pair for Raum subdivision MLP
```

This gives millions of objects with perfect camera poses at zero
computational cost for pose estimation.

### From real scans (COLMAP required)

```
Video/photos -> COLMAP (SfM) -> sparse point cloud + poses
  -> gsplat train -> .ply
```

Needed for: real building materials, custom objects, things not in Objaverse.

### Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| gsplat (nerfstudio) | GS training | 4x less memory, 15% faster than original |
| COLMAP | Camera pose estimation | Only needed for real captures |
| Blender | Multi-view rendering from meshes | Scriptable, batch-friendly |
| Nerfstudio | Full pipeline (splatfacto method) | End-to-end, good CLI |

## Recommended Data Build Plan for Raum 1.4

### Phase 1: Core categories (1 week)

Train GS reconstructions for 200-500 Objaverse objects covering:
- 50 architectural elements (towers, walls, roofs, arches, columns)
- 50 natural objects (rocks, trees, bushes, terrain patches, water)
- 50 vehicles (ships, cars, planes)
- 50 furniture (chairs, tables, beds, shelves)

Use Blender batch rendering (91 views per object), then gsplat training.
Store as `data/objaverse_gs/{category}/{object_id}.ply`.

### Phase 2: Template extraction (2-3 days)

Cluster GS reconstructions by semantic category. Extract representative
templates (centroid of each cluster, or pick highest-quality exemplar).
Store as `data/templates/{category}.ply`.

### Phase 3: Scale (ongoing)

Expand to 5,000+ objects. Prioritize categories that appear in demo
scenes. Can run as background batch job.

## Cost Estimate

| Step | Time (RTX 4090) | Storage |
|------|-----------------|---------|
| Blender render (500 objects x 91 views) | ~6 hours | ~50 GB images |
| gsplat training (500 objects, 7K iters each) | ~20 hours | ~25 GB .ply |
| Template extraction | Minutes | ~500 MB |

Total: ~1 day compute + 75 GB storage for the initial 500-object build.
