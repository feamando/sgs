"""
Build Gaussian Splatting training data from Objaverse objects.

For each selected object:
1. Download from Objaverse (if not cached)
2. Render multi-view images using trimesh (no Blender dependency)
3. Save point cloud as a GS-compatible .pt file

This produces the training data for the subdivision MLP.

Usage:
    python scripts/build_objaverse_gs.py --n-objects 500 --output data/objaverse_gs
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Priority categories for Raum scene generation
PRIORITY_CATEGORIES = [
    "tower", "wall", "roof", "column", "arch", "door", "window", "fence",
    "rock", "tree", "bush", "grass", "mountain", "hill", "cliff",
    "water", "river", "lake", "ocean",
    "ship", "boat", "car", "truck", "plane",
    "chair", "table", "bed", "lamp", "shelf",
    "house", "castle", "church", "bridge", "lighthouse",
    "sword", "shield", "flag", "barrel", "crate",
    "dragon", "horse", "bird", "fish", "dog",
]


def mesh_to_gaussian_cloud(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray | None,
    n_points: int = 2000,
) -> dict[str, torch.Tensor]:
    """
    Sample a point cloud from a mesh and create Gaussian parameters.

    Args:
        vertices: [V, 3] mesh vertices
        faces: [F, 3] face indices
        vertex_colors: [V, 3] RGB colors or None
        n_points: number of Gaussians to create

    Returns:
        dict with positions, scales, rotations, opacities, colors
    """
    # Sample points uniformly on mesh surface
    # Use face-area-weighted sampling
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Face areas
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=1) * 0.5
    areas_sum = areas.sum()
    if areas_sum < 1e-10:
        # Degenerate mesh, just sample vertices
        idx = np.random.choice(len(vertices), min(n_points, len(vertices)), replace=True)
        positions = vertices[idx]
        if vertex_colors is not None:
            colors = vertex_colors[idx]
        else:
            colors = np.ones((len(positions), 3)) * 0.5
    else:
        probs = areas / areas_sum
        face_idx = np.random.choice(len(faces), n_points, p=probs)

        # Random barycentric coordinates
        r1 = np.random.rand(n_points, 1)
        r2 = np.random.rand(n_points, 1)
        sqrt_r1 = np.sqrt(r1)
        bary = np.concatenate([1 - sqrt_r1, sqrt_r1 * (1 - r2), sqrt_r1 * r2], axis=1)

        # Interpolate positions
        sampled_v0 = vertices[faces[face_idx, 0]]
        sampled_v1 = vertices[faces[face_idx, 1]]
        sampled_v2 = vertices[faces[face_idx, 2]]
        positions = (bary[:, 0:1] * sampled_v0 +
                     bary[:, 1:2] * sampled_v1 +
                     bary[:, 2:3] * sampled_v2)

        # Interpolate colors
        if vertex_colors is not None:
            c0 = vertex_colors[faces[face_idx, 0]]
            c1 = vertex_colors[faces[face_idx, 1]]
            c2 = vertex_colors[faces[face_idx, 2]]
            colors = (bary[:, 0:1] * c0 + bary[:, 1:2] * c1 + bary[:, 2:3] * c2)
        else:
            colors = np.ones((n_points, 3)) * 0.5

    # Normalize to unit sphere
    center = positions.mean(axis=0)
    positions = positions - center
    extent = np.abs(positions).max() + 1e-6
    positions = positions / extent

    # Compute per-point scale from local density (KNN distance)
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)
    dists, _ = tree.query(positions, k=min(6, len(positions)))
    avg_nn_dist = dists[:, 1:].mean(axis=1)  # skip self
    log_scales = np.log(np.maximum(avg_nn_dist * 0.5, 1e-6))
    scales = np.stack([log_scales] * 3, axis=1)

    return {
        "positions": torch.tensor(positions, dtype=torch.float32),
        "scales": torch.tensor(scales, dtype=torch.float32),
        "rotations": torch.tensor([[1, 0, 0, 0]] * len(positions), dtype=torch.float32),
        "opacities": torch.zeros(len(positions), dtype=torch.float32),
        "colors": torch.tensor(np.clip(colors, 0, 1), dtype=torch.float32),
    }


def process_object(
    obj_path: Path,
    output_dir: Path,
    category: str,
    obj_id: str,
    n_points: int = 2000,
) -> bool:
    """Process one 3D object file into GS training data."""
    try:
        import trimesh
    except ImportError:
        print("trimesh required: pip install trimesh")
        return False

    try:
        mesh = trimesh.load(str(obj_path), force="mesh")
    except Exception:
        return False

    if not hasattr(mesh, "vertices") or len(mesh.vertices) < 10:
        return False

    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int64)

    # Extract vertex colors if available
    vertex_colors = None
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
        vc = np.array(mesh.visual.vertex_colors)
        if vc.shape[1] >= 3:
            vertex_colors = vc[:, :3].astype(np.float64) / 255.0

    data = mesh_to_gaussian_cloud(vertices, faces, vertex_colors, n_points)

    # Save
    out_dir = output_dir / category / obj_id
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_dir / "model.pt")

    meta = {
        "category": category,
        "object_id": obj_id,
        "n_gaussians": data["positions"].shape[0],
        "source_file": str(obj_path.name),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return True


def main():
    parser = argparse.ArgumentParser(description="Build GS data from Objaverse")
    parser.add_argument("--categories", default=",".join(PRIORITY_CATEGORIES[:10]),
                        help="Comma-separated category names")
    parser.add_argument("--n-objects", type=int, default=500,
                        help="Total objects to process")
    parser.add_argument("--n-points", type=int, default=2000,
                        help="Gaussians per object")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--objaverse-dir", default=None,
                        help="Local Objaverse cache directory (optional)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = [c.strip() for c in args.categories.split(",")]

    print(f"Categories: {categories}")
    print(f"Target: {args.n_objects} objects, {args.n_points} Gaussians each")
    print(f"Output: {output_dir}")

    if args.objaverse_dir:
        # Process from local directory
        obj_dir = Path(args.objaverse_dir)
        processed = 0
        mesh_extensions = ["*.obj", "*.stl", "*.gltf", "*.glb", "*.fbx", "*.ply"]

        for cat_dir in sorted(obj_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            cat = cat_dir.name
            if cat in ("game_kits", "kenney", "manifest.json"):
                continue

            # Find all mesh files in this category (including in subdirectories)
            mesh_files = []
            for ext in mesh_extensions:
                mesh_files.extend(cat_dir.rglob(ext))

            for obj_file in sorted(mesh_files):
                if processed >= args.n_objects:
                    break
                obj_id = obj_file.parent.name if obj_file.name == "scene.gltf" else obj_file.stem
                if process_object(obj_file, output_dir, cat, obj_id, args.n_points):
                    processed += 1
                    if processed % 10 == 0:
                        print(f"  Processed {processed}/{args.n_objects}")
            if processed >= args.n_objects:
                break
        print(f"Done. Processed {processed} objects.")
    else:
        try:
            import objaverse
            print("Downloading from Objaverse...")
            uids = objaverse.load_uids()
            annotations = objaverse.load_annotations(uids[:args.n_objects * 2])

            processed = 0
            per_category = args.n_objects // len(categories)

            for cat in categories:
                cat_uids = [uid for uid, ann in annotations.items()
                            if cat.lower() in str(ann.get("name", "")).lower()][:per_category]
                if not cat_uids:
                    continue

                objects = objaverse.load_objects(cat_uids)
                for uid, path in objects.items():
                    if process_object(Path(path), output_dir, cat, uid, args.n_points):
                        processed += 1
                if processed >= args.n_objects:
                    break

            print(f"Done. Processed {processed} objects.")
        except ImportError:
            print("objaverse package not installed. Options:")
            print("  1. pip install objaverse")
            print("  2. Use --objaverse-dir with a local directory of .obj files")
            sys.exit(1)


if __name__ == "__main__":
    main()
