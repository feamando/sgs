"""
ShapeNet Core v2 → Gaussian blob library for Raum 1.1.

Downloads are manual (ShapeNet requires registration at shapenet.org).
Expected layout after extraction:

    data/shapenet_core_v2/
      02691156/           ← synset id (airplane)
        <model_id>/
          models/
            model_normalized.obj
      02958343/           ← synset id (car)
        ...

Usage:
    python scripts/build_blobs_shapenet.py ^
      --shapenet-root data/shapenet_core_v2 ^
      --n-points 1000 ^
      --out-dir data/blobs

Outputs:
    data/blobs/<class_name>.pt   — {means, scales_log, opacities, colors}
    data/blobs/index.json        — ordered list of class names
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ShapeNet synset → human-readable name (subset matching Raum vocab).
# Names chosen to be single GloVe words in top-50k.
SYNSET_TO_NAME = {
    "02691156": "airplane",
    "02747177": "bin",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02871439": "bookshelf",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02942699": "camera",
    "02946921": "can",
    "02954340": "cap",
    "02958343": "car",
    "03001627": "chair",
    "03046257": "clock",
    "03085013": "keyboard",
    "03207941": "dishwasher",
    "03211117": "monitor",
    "03261776": "headphone",
    "03325088": "faucet",
    "03337140": "file",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "speaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorcycle",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04379243": "table",
    "04401088": "telephone",
    "04460130": "tower",
    "04468005": "train",
    "04530566": "washer",
    "04554684": "washer",
}

# Drop categories that are too thin for good Gaussian fit or have
# ambiguous names that collide with common English words.
_SKIP = {"rifle", "file"}

# Default: all ShapeNet categories minus skipped ones
DEFAULT_CATEGORIES = sorted(set(SYNSET_TO_NAME.values()) - _SKIP)


def parse_args():
    p = argparse.ArgumentParser(description="Build Gaussian blob library from ShapeNet")
    p.add_argument("--shapenet-root", type=str, default="data/shapenet_core_v2",
                   help="Root of ShapeNet Core v2 extraction")
    p.add_argument("--categories", type=str, default=None,
                   help="Comma-separated class names (default: 30 curated)")
    p.add_argument("--n-points", type=int, default=1000,
                   help="Gaussians per blob (surface samples)")
    p.add_argument("--out-dir", type=str, default="data/blobs",
                   help="Output directory")
    p.add_argument("--scale-radius", type=float, default=0.5,
                   help="Normalize meshes to fit in this radius")
    return p.parse_args()


def _find_synset_for_name(name: str) -> str | None:
    for synset, n in SYNSET_TO_NAME.items():
        if n == name:
            return synset
    return None


def _pick_canonical_model(synset_dir: Path) -> Path | None:
    """Pick the first model with a valid .obj file."""
    model_dirs = sorted([d for d in synset_dir.iterdir() if d.is_dir()])
    for md in model_dirs:
        obj_path = md / "models" / "model_normalized.obj"
        if obj_path.exists():
            return obj_path
    return None


def _mesh_to_gaussian_cloud(
    obj_path: Path,
    n_points: int,
    scale_radius: float,
) -> dict:
    """Load mesh, sample surface, compute Gaussian parameters."""
    import trimesh

    mesh = trimesh.load(str(obj_path), force="mesh", process=True)

    # Normalize to unit sphere then scale
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid
    max_extent = np.abs(mesh.vertices).max()
    if max_extent > 1e-6:
        mesh.vertices *= scale_radius / max_extent

    # Sample surface points
    points, face_indices = trimesh.sample.sample_surface(mesh, n_points)
    points = points.astype(np.float32)

    # Compute per-point log-scale from k-nearest-neighbor distances
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=4)  # k=4: self + 3 neighbors
    nn_dist = dists[:, 1:].mean(axis=1)  # exclude self
    nn_dist = np.clip(nn_dist, 1e-4, None)
    scales_log = np.log(nn_dist * 0.5).astype(np.float32)

    # Colors from mesh face albedo (if available)
    if mesh.visual and hasattr(mesh.visual, "face_colors"):
        face_colors = mesh.visual.face_colors[face_indices, :3] / 255.0
        colors = face_colors.astype(np.float32)
    else:
        colors = np.full((n_points, 3), 0.7, dtype=np.float32)

    return {
        "means": torch.from_numpy(points),
        "scales_log": torch.from_numpy(
            np.stack([scales_log, scales_log, scales_log], axis=-1)
        ),
        "opacities": torch.full((n_points,), 2.0),  # sigmoid(2) ~ 0.88
        "colors": torch.from_numpy(colors),
    }


def main():
    args = parse_args()

    categories = DEFAULT_CATEGORIES
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    shapenet_root = Path(args.shapenet_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not shapenet_root.exists():
        print(f"ERROR: ShapeNet root not found at {shapenet_root}")
        print(f"\nTo download ShapeNet Core v2:")
        print(f"  1. Register at https://shapenet.org")
        print(f"  2. Download ShapeNet Core v2 (~5 GB)")
        print(f"  3. Extract to {shapenet_root}")
        print(f"\nExpected structure:")
        print(f"  {shapenet_root}/02691156/  (airplane)")
        print(f"  {shapenet_root}/02958343/  (car)")
        print(f"  ...")
        sys.exit(1)

    print(f"ShapeNet root: {shapenet_root}")
    print(f"Categories: {len(categories)}")
    print(f"Points per blob: {args.n_points}")
    print(f"Output: {out_dir}")
    print()

    built = []
    failed = []

    for name in categories:
        synset = _find_synset_for_name(name)
        if synset is None:
            print(f"  SKIP {name}: no synset mapping")
            failed.append(name)
            continue

        synset_dir = shapenet_root / synset
        if not synset_dir.exists():
            print(f"  SKIP {name} ({synset}): directory not found")
            failed.append(name)
            continue

        obj_path = _pick_canonical_model(synset_dir)
        if obj_path is None:
            print(f"  SKIP {name} ({synset}): no model_normalized.obj found")
            failed.append(name)
            continue

        try:
            blob = _mesh_to_gaussian_cloud(obj_path, args.n_points, args.scale_radius)
            out_path = out_dir / f"{name}.pt"
            torch.save(blob, out_path)
            built.append(name)
            print(f"  OK   {name:12s} ({synset}) -> {blob['means'].shape[0]} gaussians")
        except Exception as e:
            print(f"  FAIL {name} ({synset}): {e}")
            failed.append(name)

    # Write index
    index_path = out_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(built, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Built: {len(built)} blobs")
    print(f"Failed/skipped: {len(failed)}")
    if failed:
        print(f"  {failed}")
    print(f"Index: {index_path}")
    print(f"\nNext: retrain the bridge with --blobs-dir {out_dir} --n-blobs-max {len(built)}")


if __name__ == "__main__":
    main()
