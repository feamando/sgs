"""
Objaverse → Gaussian blob library for Raum 1.1.

No registration required. Downloads only the specific objects needed.
By default selects up to 200 categories from Objaverse's LVIS labels,
ranked by annotation count (more annotations = higher quality meshes).

Usage:
    python scripts/build_blobs_objaverse.py --n-points 1000 --out-dir data/blobs
    python scripts/build_blobs_objaverse.py --max-classes 100 --out-dir data/blobs

Outputs:
    data/blobs/<class_name>.pt   — {means, scales_log, opacities, colors}
    data/blobs/index.json        — ordered list of class names
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Categories to skip: too thin for Gaussian fit, ambiguous words, or
# non-objects (materials, abstract concepts).
_SKIP_PATTERNS = {
    "rifle", "pistol", "sword", "gun", "arrow",  # too thin
    "person", "man", "woman", "boy", "girl", "baby",  # people
    "hand", "face", "head", "leg", "foot",  # body parts
    "water", "sky", "grass", "snow", "sand",  # materials/scenes
}


def _lvis_name_to_blob_name(lvis_name: str) -> str | None:
    """Convert LVIS category name to a clean single-word blob name."""
    # Strip parenthetical qualifiers: "car_(automobile)" → "car"
    clean = re.sub(r"_?\(.*?\)", "", lvis_name).strip("_")
    # Replace underscores with nothing for single-word check
    parts = clean.split("_")

    # Use first word if compound (e.g. "laptop_computer" → "laptop")
    # Unless first word is too generic
    if len(parts) == 1:
        name = parts[0]
    elif parts[0] in ("electric", "musical", "computer", "baby"):
        name = parts[1] if len(parts) > 1 else parts[0]
    else:
        name = parts[0]

    # Skip if in blocklist
    if name in _SKIP_PATTERNS:
        return None

    # Must be at least 3 chars and look like a noun
    if len(name) < 3:
        return None

    return name


def parse_args():
    p = argparse.ArgumentParser(description="Build blob library from Objaverse")
    p.add_argument("--max-classes", type=int, default=300,
                   help="Max categories to include (ranked by annotation count)")
    p.add_argument("--categories", type=str, default=None,
                   help="Comma-separated blob names (overrides --max-classes)")
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

    print(f"Max classes: {args.max_classes}")
    print(f"Points per blob: {args.n_points}")
    print(f"Output: {out_dir}")

    # Load LVIS annotations (small metadata download)
    print("\n=== Loading LVIS annotations ===")
    lvis_annotations = objaverse.load_lvis_annotations()
    print(f"  LVIS categories available: {len(lvis_annotations)}")

    # Build category selection: rank by annotation count, deduplicate names
    if args.categories:
        # Explicit list: find matching LVIS keys
        target_names = set(c.strip() for c in args.categories.split(","))
        lvis_to_use = {}
        for lvis_name in lvis_annotations:
            blob_name = _lvis_name_to_blob_name(lvis_name)
            if blob_name and blob_name in target_names:
                lvis_to_use[lvis_name] = blob_name
    else:
        # Auto-select top N by annotation count
        ranked = sorted(
            lvis_annotations.items(),
            key=lambda kv: len(kv[1]),
            reverse=True,
        )
        lvis_to_use = {}
        seen_names = set()
        for lvis_name, uids in ranked:
            if len(lvis_to_use) >= args.max_classes:
                break
            blob_name = _lvis_name_to_blob_name(lvis_name)
            if blob_name is None:
                continue
            if blob_name in seen_names:
                continue
            if len(uids) < 3:
                continue  # too few examples to pick a good canonical
            seen_names.add(blob_name)
            lvis_to_use[lvis_name] = blob_name

    print(f"  Selected: {len(lvis_to_use)} categories")

    # Load full annotations for canonical selection
    print("\n=== Loading object annotations ===")
    annotations = objaverse.load_annotations()

    # Select one canonical UID per category
    print("\n=== Selecting canonical objects ===")
    uid_to_category: dict[str, str] = {}
    missing = []

    for lvis_name, blob_name in sorted(lvis_to_use.items(), key=lambda kv: kv[1]):
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
