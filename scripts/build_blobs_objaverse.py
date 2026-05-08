"""
Objaverse → Gaussian blob library for Raum 1.1.

No registration required. Downloads only the specific objects needed
(~50-250 MB for 49 classes, not the full 8.9 TB dataset).

Usage:
    python scripts/build_blobs_objaverse.py --n-points 1000 --out-dir data/blobs

Outputs:
    data/blobs/<class_name>.pt   — {means, scales_log, opacities, colors}
    data/blobs/index.json        — ordered list of class names
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# LVIS category name → clean GloVe-compatible name for the blob library.
# LVIS uses compound names like "car_(automobile)"; we map to single words.
LVIS_TO_BLOB_NAME = {
    "airplane": "airplane",
    "backpack": "backpack",
    "banana": "banana",
    "basket": "basket",
    "bathtub": "bathtub",
    "bed": "bed",
    "bench": "bench",
    "bicycle": "bicycle",
    "boat": "boat",
    "book": "book",
    "bookshelf": "bookshelf",
    "bottle": "bottle",
    "bowl": "bowl",
    "bus_(vehicle)": "bus",
    "cabinet": "cabinet",
    "cake": "cake",
    "camera": "camera",
    "candle": "candle",
    "car_(automobile)": "car",
    "chair": "chair",
    "clock": "clock",
    "cup": "cup",
    "desk": "desk",
    "drum_(musical_instrument)": "drum",
    "guitar": "guitar",
    "hamburger": "hamburger",
    "hat": "hat",
    "helmet": "helmet",
    "jar": "jar",
    "keyboard_(computer)": "keyboard",
    "knife": "knife",
    "lamp": "lamp",
    "laptop_computer": "laptop",
    "microphone": "microphone",
    "microwave_oven": "microwave",
    "motorcycle": "motorcycle",
    "mug": "mug",
    "piano": "piano",
    "pillow": "pillow",
    "plate": "plate",
    "pot": "pot",
    "refrigerator": "refrigerator",
    "skateboard": "skateboard",
    "sofa": "sofa",
    "speaker_(stero_equipment)": "speaker",
    "stove": "stove",
    "table": "table",
    "telephone": "telephone",
    "television_set": "monitor",
    "toilet": "toilet",
    "tower": "tower",
    "train_(railroad_vehicle)": "train",
    "truck": "truck",
    "vase": "vase",
}


def parse_args():
    p = argparse.ArgumentParser(description="Build blob library from Objaverse")
    p.add_argument("--categories", type=str, default=None,
                   help="Comma-separated blob names to build (default: all)")
    p.add_argument("--n-points", type=int, default=1000,
                   help="Gaussians per blob (surface samples)")
    p.add_argument("--out-dir", type=str, default="data/blobs")
    p.add_argument("--scale-radius", type=float, default=0.5)
    p.add_argument("--download-processes", type=int, default=4)
    return p.parse_args()


def _pick_canonical(uids: list[str], annotations: dict) -> str:
    """Pick an object with moderate face count (good mesh, not too heavy)."""
    scored = []
    for uid in uids[:50]:  # only check first 50 to keep it fast
        if uid in annotations:
            ann = annotations[uid]
            faces = ann.get("faceCount", 0)
            if 5000 <= faces <= 50000:
                scored.append((uid, faces))
    if scored:
        scored.sort(key=lambda x: x[1])
        return scored[len(scored) // 2][0]
    return uids[0]


def _glb_to_gaussian_cloud(glb_path: str, n_points: int, scale_radius: float) -> dict:
    """Load GLB mesh, sample surface, compute Gaussian parameters."""
    import trimesh

    mesh = trimesh.load(glb_path, force="mesh", process=True)

    if mesh.vertices.shape[0] == 0:
        raise ValueError("Empty mesh")

    # Normalize
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid
    max_extent = np.abs(mesh.vertices).max()
    if max_extent > 1e-6:
        mesh.vertices *= scale_radius / max_extent

    # Sample surface
    points, face_indices = trimesh.sample.sample_surface(mesh, n_points)
    points = points.astype(np.float32)

    # Log-scale from kNN distances
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=4)
    nn_dist = dists[:, 1:].mean(axis=1)
    nn_dist = np.clip(nn_dist, 1e-4, None)
    scales_log = np.log(nn_dist * 0.5).astype(np.float32)

    # Colors from mesh
    if mesh.visual and hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None:
        try:
            face_colors = mesh.visual.face_colors[face_indices, :3] / 255.0
            colors = face_colors.astype(np.float32)
        except (IndexError, TypeError):
            colors = np.full((n_points, 3), 0.7, dtype=np.float32)
    else:
        colors = np.full((n_points, 3), 0.7, dtype=np.float32)

    return {
        "means": torch.from_numpy(points),
        "scales_log": torch.from_numpy(
            np.stack([scales_log, scales_log, scales_log], axis=-1)
        ),
        "opacities": torch.full((n_points,), 2.0),
        "colors": torch.from_numpy(colors),
    }


def main():
    args = parse_args()

    try:
        import objaverse
    except ImportError:
        print("ERROR: pip install objaverse")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter categories
    if args.categories:
        target_names = [c.strip() for c in args.categories.split(",")]
        lvis_to_use = {k: v for k, v in LVIS_TO_BLOB_NAME.items() if v in target_names}
    else:
        lvis_to_use = LVIS_TO_BLOB_NAME

    print(f"Target: {len(lvis_to_use)} categories")
    print(f"Points per blob: {args.n_points}")
    print(f"Output: {out_dir}")

    # Load LVIS annotations (small metadata download)
    print("\n=== Loading LVIS annotations ===")
    lvis_annotations = objaverse.load_lvis_annotations()
    print(f"  LVIS categories available: {len(lvis_annotations)}")

    # Load full annotations for canonical selection
    print("\n=== Loading object annotations ===")
    annotations = objaverse.load_annotations()

    # Select one canonical UID per category
    print("\n=== Selecting canonical objects ===")
    uid_to_category: dict[str, str] = {}
    missing = []

    for lvis_name, blob_name in sorted(lvis_to_use.items()):
        if lvis_name not in lvis_annotations or len(lvis_annotations[lvis_name]) == 0:
            print(f"  SKIP {blob_name}: no LVIS annotations for '{lvis_name}'")
            missing.append(blob_name)
            continue
        uid = _pick_canonical(lvis_annotations[lvis_name], annotations)
        uid_to_category[uid] = blob_name
        print(f"  {blob_name:12s} <- {uid}")

    if not uid_to_category:
        print("ERROR: no objects to download")
        sys.exit(1)

    # Download GLB files
    print(f"\n=== Downloading {len(uid_to_category)} objects ===")
    objects = objaverse.load_objects(
        uids=list(uid_to_category.keys()),
        download_processes=args.download_processes,
    )

    # Convert to Gaussian clouds
    print("\n=== Converting to Gaussian blobs ===")
    built = []
    failed = []

    for uid, glb_path in objects.items():
        blob_name = uid_to_category[uid]
        try:
            blob = _glb_to_gaussian_cloud(glb_path, args.n_points, args.scale_radius)
            out_path = out_dir / f"{blob_name}.pt"
            torch.save(blob, out_path)
            built.append(blob_name)
            print(f"  OK   {blob_name:12s} -> {blob['means'].shape[0]} gaussians")
        except Exception as e:
            print(f"  FAIL {blob_name}: {e}")
            failed.append(blob_name)

    # Write index
    index_path = out_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(sorted(built), f, indent=2)

    print(f"\n{'='*50}")
    print(f"Built: {len(built)} blobs")
    print(f"Failed: {len(failed)}")
    print(f"Missing LVIS: {len(missing)}")
    if failed:
        print(f"  Failed: {failed}")
    if missing:
        print(f"  Missing: {missing}")
    print(f"Index: {index_path}")
    print(f"\nNext: retrain the bridge with --blobs-dir {out_dir}")


if __name__ == "__main__":
    main()
